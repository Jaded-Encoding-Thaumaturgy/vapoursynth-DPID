#include "VapourSynth4.h"
#include "VSHelper4.h"
#include <cstdint>
#include <cmath>
#include <string>
#include <algorithm>
#include <memory>


struct DpidData {
    VSNode *node1, *node2;
    int dst_w, dst_h;
    float lambda[3];
    float src_left[3], src_top[3];
    float src_width[3], src_height[3];
    bool process[3];
    bool read_chromaloc;
};


static float contribution(float f, float x, float y, 
    float sx, float ex, float sy, float ey) {

    if (x < sx)
        f *= 1.0f - (sx - x);

    if ((x + 1.0f) > ex)
        f *= ex - x;

    if (y < sy)
        f *= 1.0f - (sy - y);

    if ((y + 1.0f) > ey)
        f *= ey - y;

    return f;
}

template<typename T>
static void dpidProcess(const T * VS_RESTRICT srcp, int src_stride, 
    const T * VS_RESTRICT downp, int down_stride, 
    T * VS_RESTRICT dstp, int dst_stride, 
    int src_w, int src_h, int dst_w, int dst_h, float lambda, 
    float src_left, float src_top, float src_width, float src_height) {

    const float scale_x = src_width / dst_w;
    const float scale_y = src_height / dst_h;

    for (int outer_y = 0; outer_y < dst_h; ++outer_y) {
        for (int outer_x = 0; outer_x < dst_w; ++outer_x) {

            // avg = RemoveGrain(down, 11)
            float avg {};
            for (int inner_y = -1; inner_y <= 1; ++inner_y) {
                for (int inner_x = -1; inner_x <= 1; ++inner_x) {

                    int y = std::clamp(outer_y + inner_y, 0, dst_h - 1);
                    int x = std::clamp(outer_x + inner_x, 0, dst_w - 1);

                    T pixel = downp[y * down_stride + x];
                    avg += pixel * (2 - std::abs(inner_y)) * (2 - std::abs(inner_x));
                }
            }
            avg /= 16.f;

            // Dpid
            const float sx = std::clamp(outer_x * scale_x + src_left, 0.f, static_cast<float>(src_w));
            const float ex = std::clamp((outer_x + 1) * scale_x + src_left, 0.f, static_cast<float>(src_w));
            const float sy = std::clamp(outer_y * scale_y + src_top, 0.f, static_cast<float>(src_h));
            const float ey = std::clamp((outer_y + 1) * scale_y + src_top, 0.f, static_cast<float>(src_h));

            const int sxr = static_cast<int>(std::floor(sx));
            const int exr = static_cast<int>(std::ceil(ex));
            const int syr = static_cast<int>(std::floor(sy));
            const int eyr = static_cast<int>(std::ceil(ey));

            float sum_pixel {};
            float sum_weight {};

            for (int inner_y = syr; inner_y < eyr; ++inner_y) {
                for (int inner_x = sxr; inner_x < exr; ++inner_x) {
                    T pixel = srcp[inner_y * src_stride + inner_x];
                    float distance = std::abs(avg - static_cast<float>(pixel));
                    float weight = std::pow(distance, lambda);
                    weight = contribution(weight, static_cast<float>(inner_x), static_cast<float>(inner_y), sx, ex, sy, ey);

                    sum_pixel += weight * pixel;
                    sum_weight += weight;
                }
            }

            dstp[outer_y * dst_stride + outer_x] = static_cast<T>((sum_weight == 0.f) ? avg : sum_pixel / sum_weight);
        }
    }
}

static const VSFrame *VS_CC dpidGetframe(int n, int activationReason, void *instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
    DpidData *d = reinterpret_cast<DpidData *>(instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node1, frameCtx);
        vsapi->requestFrameFilter(n, d->node2, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrame *src1 = vsapi->getFrameFilter(n, d->node1, frameCtx);
        const VSFrame *src2 = vsapi->getFrameFilter(n, d->node2, frameCtx);
        const VSVideoFormat *fi = vsapi->getVideoFrameFormat(src2);

        const VSFrame * fr[] = {
            d->process[0] ? nullptr : src2, 
            d->process[1] ? nullptr : src2, 
            d->process[2] ? nullptr : src2};

        constexpr int pl[] = {0, 1, 2};

        VSFrame *dst = vsapi->newVideoFrame2(
            fi, vsapi->getFrameWidth(src2, 0), vsapi->getFrameHeight(src2, 0), fr, pl, src2, core);

        for (int plane = 0; plane < fi->numPlanes; ++plane) {
            if (d->process[plane]) {
                const void *src1p = vsapi->getReadPtr(src1, plane);
                const int src1_stride = vsapi->getStride(src1, plane) / fi->bytesPerSample;
                const void *src2p = vsapi->getReadPtr(src2, plane);
                const int src2_stride = vsapi->getStride(src2, plane) / fi->bytesPerSample;
                void *dstp = vsapi->getWritePtr(dst, plane);
                const int dst_stride = vsapi->getStride(dst, plane) / fi->bytesPerSample;

                const int src_w = vsapi->getFrameWidth(src1, plane);
                const int src_h = vsapi->getFrameHeight(src1, plane);
                const int dst_w = vsapi->getFrameWidth(src2, plane);
                const int dst_h = vsapi->getFrameHeight(src2, plane);

                const float hSubS = plane == 0 ? 1.0f : static_cast<float>(1 << fi->subSamplingW);
                const float vSubS = plane == 0 ? 1.0f : static_cast<float>(1 << fi->subSamplingH);

                float src_left, src_top;

                float src_width = d->src_width[plane] / hSubS;
                if (src_width == 0.0f)
                    src_width = static_cast<float>(src_w);
                float src_height = d->src_height[plane] / vSubS;
                if (src_height == 0.0f)
                    src_height = static_cast<float>(src_h);

                if (plane != 0 && d->read_chromaloc) { 
                    int chromaLocation;

                    {
                        int err;

                        chromaLocation = vsh::int64ToIntS(vsapi->mapGetInt(vsapi->getFramePropertiesRO(src2), "_ChromaLocation", 0, &err));
                        if (err) {
                            chromaLocation = 0;
                        }
                    }

                    const float hCPlace = (chromaLocation == 0 || chromaLocation == 2 || chromaLocation == 4) 
                        ? (0.5f - hSubS / 2) : 0.f;
                    const float hScale = static_cast<float>(dst_w) / src_width;

                    const float vCPlace = (chromaLocation == 2 || chromaLocation == 3) 
                        ? (0.5f - vSubS / 2) : ((chromaLocation == 4 || chromaLocation == 5) ? (vSubS / 2 - 0.5f) : 0.f);
                    const float vScale = static_cast<float>(dst_h) / src_height;

                    src_left = ((d->src_left[plane] - hCPlace) * hScale + hCPlace) / hScale / hSubS;
                    src_top = ((d->src_top[plane] - vCPlace) * vScale + vCPlace) / vScale / vSubS;
                } else {
                    src_left = d->src_left[plane];
                    src_top = d->src_top[plane];
                }

                // process
                if (fi->sampleType == stInteger) {
                    if (fi->bytesPerSample == 1) {
                        dpidProcess(
                            reinterpret_cast<const uint8_t *>(src1p), src1_stride, 
                            reinterpret_cast<const uint8_t *>(src2p), src2_stride, 
                            reinterpret_cast<uint8_t *>(dstp), dst_stride, 
                            src_w, src_h, dst_w, dst_h, d->lambda[plane], 
                            src_left, src_top, src_width, src_height);
                    } else if (fi->bytesPerSample == 2) {
                        dpidProcess(
                            reinterpret_cast<const uint16_t *>(src1p), src1_stride, 
                            reinterpret_cast<const uint16_t *>(src2p), src2_stride, 
                            reinterpret_cast<uint16_t *>(dstp), dst_stride, 
                            src_w, src_h, dst_w, dst_h, d->lambda[plane], 
                            src_left, src_top, src_width, src_height);
                    }
                } else if (fi->sampleType == stFloat) {
                    if (fi->bytesPerSample == 4) {
                        dpidProcess(
                            reinterpret_cast<const float *>(src1p), src1_stride, 
                            reinterpret_cast<const float *>(src2p), src2_stride, 
                            reinterpret_cast<float *>(dstp), dst_stride, 
                            src_w, src_h, dst_w, dst_h, d->lambda[plane], 
                            src_left, src_top, src_width, src_height);
                    }
                }
            }
        }

        vsapi->freeFrame(src1);
        vsapi->freeFrame(src2);
        return dst;
    }

    return nullptr;
}

static void VS_CC dpidNodeFree(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    DpidData *d = reinterpret_cast<DpidData *>(instanceData);

    vsapi->freeNode(d->node1);
    vsapi->freeNode(d->node2);
    delete d;
}


static void VS_CC dpidRawCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    std::unique_ptr<DpidData> d = std::make_unique<DpidData>();

    d->node1 = vsapi->mapGetNode(in, "clip", 0, nullptr);
    d->node2 = vsapi->mapGetNode(in, "clip2", 0, nullptr);
    const VSVideoInfo *vi = vsapi->getVideoInfo(d->node2);
    d->dst_w = vi->width;
    d->dst_h = vi->height;

    int err;

    try {
        if (!vsh::isConstantVideoFormat(vi) ||
            (vi->format.sampleType == stInteger && vi->format.bitsPerSample > 16) ||
            (vi->format.sampleType == stFloat && vi->format.bitsPerSample != 32))
            throw std::string{"only constant format 8-16 bit integer and 32 bit float input supported"};


        if (const VSVideoInfo *vi_src = vsapi->getVideoInfo(d->node1);
            !vsh::isSameVideoFormat(&vi->format, &vi_src->format) || (vi->numFrames != vi_src->numFrames))
            throw std::string{"\"clip\" and \"clip2\" must be of the same format and number of frames"};

        const int numLambda = vsapi->mapNumElements(in, "lambda");
        if (numLambda > vi->format.numPlanes)
            throw std::string{"more \"lambda\" given than there are planes"};

        for (int i = 0; i < 3; i++) {
            if (i < numLambda)
                d->lambda[i] = static_cast<float>(vsapi->mapGetFloat(in, "lambda", i, nullptr));
            else if (i == 0)
                d->lambda[0] = 1.0f;
            else 
                d->lambda[i] = d->lambda[i-1];
        }

        const int numPlanes = vsapi->mapNumElements(in, "planes");

        for (int i = 0; i < 3; i++)
            d->process[i] = (numPlanes <= 0);

        for (int i = 0; i < numPlanes; i++) {
            const int n = vsh::int64ToIntS(vsapi->mapGetInt(in, "planes", i, nullptr));

            if (n < 0 || n >= vi->format.numPlanes)
                throw std::string{"plane index out of range"};

            if (d->process[n])
                throw std::string{"plane specified twice"};

            d->process[n] = true;
        }

        const int numSrcLeft = vsapi->mapNumElements(in, "src_left");
        if (numSrcLeft > vi->format.numPlanes)
            throw std::string{"more \"src_left\" given than there are planes"};

        const int numSrcTop = vsapi->mapNumElements(in, "src_top");
        if (numSrcTop > vi->format.numPlanes)
            throw std::string{"more \"src_top\" given than there are planes"};

        const int numSrcWidth = vsapi->mapNumElements(in, "src_width");
        if (numSrcWidth > vi->format.numPlanes)
            throw std::string{"more \"src_width\" given than there are planes"};

        const int numSrcHeight = vsapi->mapNumElements(in, "src_height");
        if (numSrcHeight > vi->format.numPlanes)
            throw std::string{"more \"src_height\" given than there are planes"};

        for (int i = 0; i < 3; i++) {
            if (i < numSrcLeft)
                d->src_left[i] = static_cast<float>(vsapi->mapGetFloat(in, "src_left", i, nullptr));
            else if (i == 0)
                d->src_left[0] = 0.0f;
            else 
                d->src_left[i] = d->src_left[i-1];

            if (i < numSrcTop)
                d->src_top[i] = static_cast<float>(vsapi->mapGetFloat(in, "src_top", i, nullptr));
            else if (i == 0)
                d->src_top[0] = 0.0f;
            else 
                d->src_top[i] = d->src_top[i-1];

            if (i < numSrcWidth) {
                d->src_width[i] = static_cast<float>(vsapi->mapGetFloat(in, "src_width", i, nullptr));
                if (d->src_width[i] < 0.0f)
                    throw std::string{"active window set by \"src_width\" must be positive"};
            } else if (i == 0)
                d->src_width[0] = 0.0f;
            else
                d->src_width[i] = d->src_width[i - 1];

            if (i < numSrcHeight) {
                d->src_height[i] = static_cast<float>(vsapi->mapGetFloat(in, "src_height", i, nullptr));
                if (d->src_height[i] < 0.0f)
                    throw std::string{"active window set by \"src_height\" must be positive"};
            } else if (i == 0)
                d->src_height[0] = 0.0f;
            else
                d->src_height[i] = d->src_height[i - 1];
        }

        d->read_chromaloc = static_cast<bool>(vsapi->mapGetInt(in, "read_chromaloc", 0, &err));

        if (err) {
            d->read_chromaloc = true;
        }

    } catch (const std::string &error) {
        vsapi->mapSetError(out, ("DpidRaw: " + error).c_str());
        vsapi->freeNode(d->node1);
        vsapi->freeNode(d->node2);
        return;
    }

    VSFilterDependency deps[] = {
        {d->node1, rpStrictSpatial},
        {d->node2, rpStrictSpatial},
    };
    
    vsapi->createVideoFilter(out, "DpidRaw", vsapi->getVideoInfo(d->node2), dpidGetframe, dpidNodeFree, fmParallel, deps, 2, d.get(), core);
    d.release();
}


static void VS_CC dpidCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    std::unique_ptr<DpidData> d = std::make_unique<DpidData>();

    VSNode *node = vsapi->mapGetNode(in, "clip", 0, nullptr);
    d->node1 = node;

    const VSVideoInfo *vi = vsapi->getVideoInfo(node);

    int err;

    try {
        if (!vsh::isConstantVideoFormat(vi) ||
            (vi->format.sampleType == stInteger && vi->format.bitsPerSample > 16) ||
            (vi->format.sampleType == stFloat && vi->format.bitsPerSample != 32))
            throw std::string{"only constant format 8-16 bit integer and 32 bit float input supported"};

        // read arguments
        d->dst_w = vsh::int64ToIntS(vsapi->mapGetInt(in, "width", 0, &err));
        if (err) {
            d->dst_w = vi->width;
        }

        d->dst_h = vsh::int64ToIntS(vsapi->mapGetInt(in, "height", 0, &err));
        if (err) {
            d->dst_h = vi->height;
        }

        if (d->dst_w == 0 && d->dst_h == 0)
            throw std::string{"\"width\" and \"height\" can not be equal to 0 at the same time"};
        else if (d->dst_w == 0)
            d->dst_w = vi->width * d->dst_h / vi->height;
        else if (d->dst_h == 0)
            d->dst_h = vi->height * d->dst_w / vi->width;

        if (d->dst_w == vi->width && d->dst_h == vi->height) 
            throw std::string{"dimensions of output is identical to input. "
                "Please consider remove the function call"};

        const int numLambda = vsapi->mapNumElements(in, "lambda");
        if (numLambda > vi->format.numPlanes)
            throw std::string{"more \"lambda\" given than there are planes"};

        for (int i = 0; i < 3; i++) {
            if (i < numLambda)
                d->lambda[i] = static_cast<float>(vsapi->mapGetFloat(in, "lambda", i, nullptr));
            else if (i == 0)
                d->lambda[0] = 1.0f;
            else 
                d->lambda[i] = d->lambda[i-1];
        }

        for (int i = 0; i < 3; i++)
            d->process[i] = true;

        const int numSrcLeft = vsapi->mapNumElements(in, "src_left");
        if (numSrcLeft > vi->format.numPlanes)
            throw std::string{"more \"src_left\" given than there are planes"};

        const int numSrcTop = vsapi->mapNumElements(in, "src_top");
        if (numSrcTop > vi->format.numPlanes)
            throw std::string{"more \"src_top\" given than there are planes"};

        const int numSrcWidth = vsapi->mapNumElements(in, "src_width");
        if (numSrcWidth > vi->format.numPlanes)
            throw std::string{"more \"src_width\" given than there are planes"};

        const int numSrcHeight = vsapi->mapNumElements(in, "src_height");
        if (numSrcHeight > vi->format.numPlanes)
            throw std::string{"more \"src_height\" given than there are planes"};

        for (int i = 0; i < 3; i++) {
            if (i < numSrcLeft)
                d->src_left[i] = static_cast<float>(vsapi->mapGetFloat(in, "src_left", i, nullptr));
            else if (i == 0)
                d->src_left[0] = 0.0f;
            else 
                d->src_left[i] = d->src_left[i-1];

            if (i < numSrcTop)
                d->src_top[i] = static_cast<float>(vsapi->mapGetFloat(in, "src_top", i, nullptr));
            else if (i == 0)
                d->src_top[0] = 0.0f;
            else 
                d->src_top[i] = d->src_top[i-1];

            if (i < numSrcWidth) {
                d->src_width[i] = static_cast<float>(vsapi->mapGetFloat(in, "src_width", i, nullptr));
                if (d->src_width[i] < 0.0f)
                    throw std::string{"active window set by \"src_width\" must be positive"};
            } else if (i == 0)
                d->src_width[0] = 0.0f;
            else
                d->src_width[i] = d->src_width[i - 1];

            if (i < numSrcHeight) {
                d->src_height[i] = static_cast<float>(vsapi->mapGetFloat(in, "src_height", i, nullptr));
                if (d->src_height[i] < 0.0f)
                    throw std::string{"active window set by \"src_height\" must be positive"};
            } else if (i == 0)
                d->src_height[0] = 0.0f;
            else
                d->src_height[i] = d->src_height[i - 1];
        }

        d->read_chromaloc = static_cast<bool>(vsapi->mapGetInt(in, "read_chromaloc", 0, &err));
        if (err) {
            d->read_chromaloc = true;
        }

        // preprocess
        VSMap * vtmp1 = vsapi->createMap();
        vsapi->mapSetNode(vtmp1, "clip", node, maReplace);
        vsapi->mapSetInt(vtmp1, "width", d->dst_w, maReplace);
        vsapi->mapSetInt(vtmp1, "height", d->dst_h, maReplace);
        vsapi->mapSetFloat(vtmp1, "src_left", d->src_left[0], maReplace);
        vsapi->mapSetFloat(vtmp1, "src_top", d->src_top[0], maReplace);
        if (d->src_width[0] != 0.0f)
            vsapi->mapSetFloat(vtmp1, "src_width", d->src_width[0], maReplace);
        if (d->src_height[0] != 0.0f)
            vsapi->mapSetFloat(vtmp1, "src_height", d->src_height[0], maReplace);

        VSMap * vtmp2 = vsapi->invoke(vsapi->getPluginByNamespace("resize", core), "Bilinear", vtmp1);
        if (vsapi->mapGetError(vtmp2)) {
            vsapi->mapSetError(out, vsapi->mapGetError(vtmp2));
            vsapi->freeMap(vtmp1);
            vsapi->freeMap(vtmp2);
            return;
        }
        vsapi->freeMap(vtmp1);

        node = vsapi->mapGetNode(vtmp2, "clip", 0, nullptr);

        d->node2 = node;

        VSFilterDependency deps[] = {
            {d->node1, rpStrictSpatial},
            {d->node2, rpStrictSpatial},
        };
        
        vsapi->createVideoFilter(out, "DpidRaw", vsapi->getVideoInfo(d->node2), dpidGetframe, dpidNodeFree, fmParallel, deps, 2, d.get(), core);
        d.release();

        vsapi->freeMap(vtmp2);
    } catch (const std::string &error) {
        vsapi->mapSetError(out, ("Dpid: " + error).c_str());
        vsapi->freeNode(node);
        return;
    }
}


VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin *plugin, const VSPLUGINAPI *vspapi) {
    vspapi->configPlugin("com.wolframrhodium.dpid", "dpid", "Rapid, Detail-Preserving Image Downscaling", VS_MAKE_VERSION(1, 0), VAPOURSYNTH_API_VERSION, 0, plugin);

    vspapi->registerFunction("DpidRaw", 
        "clip:vnode;"
        "clip2:vnode;"
        "lambda:float[]:opt;"
        "src_left:float[]:opt;"
        "src_top:float[]:opt;"
        "src_width:float[]:opt;"
        "src_height:float[]:opt;"
        "read_chromaloc:int:opt;"
        "planes:int[]:opt;", 
        "clip:vnode;", dpidRawCreate, 0, plugin);

    vspapi->registerFunction("Dpid", 
        "clip:vnode;"
        "width:int:opt;"
        "height:int:opt;"
        "lambda:float[]:opt;"
        "src_left:float[]:opt;"
        "src_top:float[]:opt;"
        "src_width:float[]:opt;"
        "src_height:float[]:opt;"
        "read_chromaloc:int:opt;",
        "clip:vnode;", dpidCreate, 0, plugin);
}
