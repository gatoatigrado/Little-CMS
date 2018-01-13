//---------------------------------------------------------------------------------
//
//  Little Color Management System
//  Copyright (c) 1998-2017 Marti Maria Saguer
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software
// is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
//---------------------------------------------------------------------------------
//

#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>
#include <immintrin.h>
#include <lcms2_plugin.h>

extern "C" {
#include "lcms2_internal.h"
}

// Transformations stuff
// -----------------------------------------------------------------------

#define DEFAULT_OBSERVER_ADAPTATION_STATE 1.0

// The Context0 observer adaptation state.
_cmsAdaptationStateChunkType _cmsAdaptationStateChunk = { DEFAULT_OBSERVER_ADAPTATION_STATE };

// Init and duplicate observer adaptation state
void _cmsAllocAdaptationStateChunk(struct _cmsContext_struct* ctx, 
                                   const struct _cmsContext_struct* src)
{
    static _cmsAdaptationStateChunkType AdaptationStateChunk = { DEFAULT_OBSERVER_ADAPTATION_STATE };
    void* from;
     
    if (src != NULL) {
        from = src ->chunks[AdaptationStateContext];       
    }
    else {
       from = &AdaptationStateChunk;
    }
    
    ctx ->chunks[AdaptationStateContext] = _cmsSubAllocDup(ctx ->MemPool, from, sizeof(_cmsAdaptationStateChunkType));     
}


// Sets adaptation state for absolute colorimetric intent in the given context.  Adaptation state applies on all 
// but cmsCreateExtendedTransformTHR().  Little CMS can handle incomplete adaptation states.
cmsFloat64Number CMSEXPORT cmsSetAdaptationStateTHR(cmsContext ContextID, cmsFloat64Number d)
{
    cmsFloat64Number prev;
    _cmsAdaptationStateChunkType* ptr = (_cmsAdaptationStateChunkType*) _cmsContextGetClientChunk(ContextID, AdaptationStateContext);

    // Get previous value for return
    prev = ptr ->AdaptationState;

    // Set the value if d is positive or zero
    if (d >= 0.0) {

        ptr ->AdaptationState = d;
    }

    // Always return previous value
    return prev;
}


// The adaptation state may be defaulted by this function. If you don't like it, use the extended transform routine
cmsFloat64Number CMSEXPORT cmsSetAdaptationState(cmsFloat64Number d)
{    
    return cmsSetAdaptationStateTHR(NULL, d);
}

// -----------------------------------------------------------------------

// Alarm codes for 16-bit transformations, because the fixed range of containers there are
// no values left to mark out of gamut. 

#define DEFAULT_ALARM_CODES_VALUE {0x7F00, 0x7F00, 0x7F00, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}

_cmsAlarmCodesChunkType _cmsAlarmCodesChunk = { DEFAULT_ALARM_CODES_VALUE };

// Sets the codes used to mark out-out-gamut on Proofing transforms for a given context. Values are meant to be 
// encoded in 16 bits.
void CMSEXPORT cmsSetAlarmCodesTHR(cmsContext ContextID, const cmsUInt16Number AlarmCodesP[cmsMAXCHANNELS])
{
    _cmsAlarmCodesChunkType* ContextAlarmCodes = (_cmsAlarmCodesChunkType*) _cmsContextGetClientChunk(ContextID, AlarmCodesContext);
       
    _cmsAssert(ContextAlarmCodes != NULL); // Can't happen
    
    memcpy(ContextAlarmCodes->AlarmCodes, AlarmCodesP, sizeof(ContextAlarmCodes->AlarmCodes));    
}

// Gets the current codes used to mark out-out-gamut on Proofing transforms for the given context.
// Values are meant to be encoded in 16 bits.
void CMSEXPORT cmsGetAlarmCodesTHR(cmsContext ContextID, cmsUInt16Number AlarmCodesP[cmsMAXCHANNELS])
{
    _cmsAlarmCodesChunkType* ContextAlarmCodes = (_cmsAlarmCodesChunkType*) _cmsContextGetClientChunk(ContextID, AlarmCodesContext);

    _cmsAssert(ContextAlarmCodes != NULL); // Can't happen

    memcpy(AlarmCodesP, ContextAlarmCodes->AlarmCodes, sizeof(ContextAlarmCodes->AlarmCodes));
}

void CMSEXPORT cmsSetAlarmCodes(const cmsUInt16Number NewAlarm[cmsMAXCHANNELS])
{
    _cmsAssert(NewAlarm != NULL);

    cmsSetAlarmCodesTHR(NULL, NewAlarm);
}

void CMSEXPORT cmsGetAlarmCodes(cmsUInt16Number OldAlarm[cmsMAXCHANNELS])
{ 
    _cmsAssert(OldAlarm != NULL);
    cmsGetAlarmCodesTHR(NULL, OldAlarm);
}


// Init and duplicate alarm codes
void _cmsAllocAlarmCodesChunk(struct _cmsContext_struct* ctx, 
                              const struct _cmsContext_struct* src)
{
    static _cmsAlarmCodesChunkType AlarmCodesChunk = { DEFAULT_ALARM_CODES_VALUE };
    void* from;
     
    if (src != NULL) {
        from = src ->chunks[AlarmCodesContext];       
    }
    else {
       from = &AlarmCodesChunk;
    }
    
    ctx ->chunks[AlarmCodesContext] = _cmsSubAllocDup(ctx ->MemPool, from, sizeof(_cmsAlarmCodesChunkType));     
}

// -----------------------------------------------------------------------

// Get rid of transform resources
void CMSEXPORT cmsDeleteTransform(cmsHTRANSFORM hTransform)
{
    _cmsTRANSFORM* p = (_cmsTRANSFORM*) hTransform;

    _cmsAssert(p != NULL);

    if (p -> GamutCheck)
        cmsPipelineFree(p -> GamutCheck);

    if (p -> Lut)
        cmsPipelineFree(p -> Lut);

    if (p ->InputColorant)
        cmsFreeNamedColorList(p ->InputColorant);

    if (p -> OutputColorant)
        cmsFreeNamedColorList(p ->OutputColorant);

    if (p ->Sequence)
        cmsFreeProfileSequenceDescription(p ->Sequence);

    if (p ->UserData)
        p ->FreeUserData(p ->ContextID, p ->UserData);

    _cmsFree(p ->ContextID, (void *) p);
}

// Apply transform.
void CMSEXPORT cmsDoTransform(cmsHTRANSFORM  Transform,
                              const void* InputBuffer,
                              void* OutputBuffer,
                              cmsUInt32Number Size)

{
    _cmsTRANSFORM* p = (_cmsTRANSFORM*) Transform;
    cmsStride stride;

    stride.BytesPerLineIn = 0;  // Not used
    stride.BytesPerLineOut = 0;
    stride.BytesPerPlaneIn = Size;
    stride.BytesPerPlaneOut = Size;
           
    p -> xform(p, InputBuffer, OutputBuffer, Size, 1, &stride);
}


// This is a legacy stride for planar
void CMSEXPORT cmsDoTransformStride(cmsHTRANSFORM  Transform,
                              const void* InputBuffer,
                              void* OutputBuffer,
                              cmsUInt32Number Size, cmsUInt32Number Stride)

{
    _cmsTRANSFORM* p = (_cmsTRANSFORM*) Transform;
    cmsStride stride;

    stride.BytesPerLineIn = 0;  
    stride.BytesPerLineOut = 0;
    stride.BytesPerPlaneIn = Stride;
    stride.BytesPerPlaneOut = Stride;

    p -> xform(p, InputBuffer, OutputBuffer, Size, 1, &stride);
}

// This is the "fast" function for plugins
void CMSEXPORT cmsDoTransformLineStride(cmsHTRANSFORM  Transform,
                              const void* InputBuffer,
                              void* OutputBuffer,
                              cmsUInt32Number PixelsPerLine,
                              cmsUInt32Number LineCount,
                              cmsUInt32Number BytesPerLineIn,
                              cmsUInt32Number BytesPerLineOut,
                              cmsUInt32Number BytesPerPlaneIn,
                              cmsUInt32Number BytesPerPlaneOut)

{
    _cmsTRANSFORM* p = (_cmsTRANSFORM*) Transform;
    cmsStride stride;

    stride.BytesPerLineIn = BytesPerLineIn;
    stride.BytesPerLineOut = BytesPerLineOut;
    stride.BytesPerPlaneIn = BytesPerPlaneIn;
    stride.BytesPerPlaneOut = BytesPerPlaneOut;

    p->xform(p, InputBuffer, OutputBuffer, PixelsPerLine, LineCount, &stride);
}



// Transform routines ----------------------------------------------------------------------------------------------------------

// Float xform converts floats. Since there are no performance issues, one routine does all job, including gamut check.
// Note that because extended range, we can use a -1.0 value for out of gamut in this case.
static
void FloatXFORM(_cmsTRANSFORM* p,
                const void* in,
                void* out, 
                cmsUInt32Number PixelsPerLine,
                cmsUInt32Number LineCount,
                const cmsStride* Stride)
{
    cmsUInt8Number* accum;
    cmsUInt8Number* output;
    cmsFloat32Number fIn[cmsMAXCHANNELS], fOut[cmsMAXCHANNELS];
    cmsFloat32Number OutOfGamut;
    cmsUInt32Number i, j, c, strideIn, strideOut;

    _cmsHandleExtraChannels(p, in, out, PixelsPerLine, LineCount, Stride);

    strideIn = 0;
    strideOut = 0;
    memset(fIn, 0, sizeof(fIn));
    memset(fOut, 0, sizeof(fIn));

    for (i = 0; i < LineCount; i++) {

        accum = (cmsUInt8Number*)in + strideIn;
        output = (cmsUInt8Number*)out + strideOut;

        for (j = 0; j < PixelsPerLine; j++) {

            accum = p->FromInputFloat(p, fIn, accum, Stride->BytesPerPlaneIn);

            // Any gamut chack to do?
            if (p->GamutCheck != NULL) {

                // Evaluate gamut marker.
                cmsPipelineEvalFloat(fIn, &OutOfGamut, p->GamutCheck);

                // Is current color out of gamut?
                if (OutOfGamut > 0.0) {

                    // Certainly, out of gamut
                    for (c = 0; c < cmsMAXCHANNELS; c++)
                        fOut[c] = -1.0;

                }
                else {
                    // No, proceed normally
                    cmsPipelineEvalFloat(fIn, fOut, p->Lut);
                }
            }
            else {

                // No gamut check at all
                cmsPipelineEvalFloat(fIn, fOut, p->Lut);
            }


            output = p->ToOutputFloat(p, fOut, output, Stride->BytesPerPlaneOut);
        }

        strideIn += Stride->BytesPerLineIn;
        strideOut += Stride->BytesPerLineOut;
    }

}


static
void NullFloatXFORM(_cmsTRANSFORM* p,
                    const void* in,
                    void* out, 
                    cmsUInt32Number PixelsPerLine,
                    cmsUInt32Number LineCount,
                    const cmsStride* Stride)

{
    cmsUInt8Number* accum;
    cmsUInt8Number* output;
    cmsFloat32Number fIn[cmsMAXCHANNELS];
    cmsUInt32Number i, j, strideIn, strideOut;

    _cmsHandleExtraChannels(p, in, out, PixelsPerLine, LineCount, Stride);

    strideIn = 0;
    strideOut = 0;
    memset(fIn, 0, sizeof(fIn));

    for (i = 0; i < LineCount; i++) {

           accum = (cmsUInt8Number*) in + strideIn;
           output = (cmsUInt8Number*) out + strideOut;

           for (j = 0; j < PixelsPerLine; j++) {

                  accum = p->FromInputFloat(p, fIn, accum, Stride ->BytesPerPlaneIn);
                  output = p->ToOutputFloat(p, fIn, output, Stride->BytesPerPlaneOut);
           }

           strideIn += Stride->BytesPerLineIn;
           strideOut += Stride->BytesPerLineOut;
    }
}

// 16 bit precision -----------------------------------------------------------------------------------------------------------

// Null transformation, only applies formatters. No cache
static
void NullXFORM(_cmsTRANSFORM* p,
               const void* in,
               void* out,
               cmsUInt32Number PixelsPerLine,
               cmsUInt32Number LineCount,
               const cmsStride* Stride)
{
    cmsUInt8Number* accum;
    cmsUInt8Number* output;
    cmsUInt16Number wIn[cmsMAXCHANNELS];
    cmsUInt32Number i, j, strideIn, strideOut;

    _cmsHandleExtraChannels(p, in, out, PixelsPerLine, LineCount, Stride);

    strideIn = 0;
    strideOut = 0;
    memset(wIn, 0, sizeof(wIn));

    for (i = 0; i < LineCount; i++) {

           accum = (cmsUInt8Number*)in + strideIn;
           output = (cmsUInt8Number*)out + strideOut;

           for (j = 0; j < PixelsPerLine; j++) {

                  accum = p->FromInput(p, wIn, accum, Stride->BytesPerPlaneIn);
                  output = p->ToOutput(p, wIn, output, Stride->BytesPerPlaneOut);
    }

           strideIn += Stride->BytesPerLineIn;
           strideOut += Stride->BytesPerLineOut;
    }

}

constexpr size_t kFasterStageBatchSize = 128;
using ChannelBatchBuffer = __attribute__((aligned(32))) float[kFasterStageBatchSize];

/** Basic API for faster stages */
class FasterStage {
public:
    explicit FasterStage(cmsStage base) : base_(base), output_buffer_(base.OutputChannels) {
        for (size_t i = 0; i < base.OutputChannels; ++i) {
            memset(output_buffer_[i], 0, kFasterStageBatchSize * sizeof(float));
        }
    }

    // Method for subclasses to implement.
    virtual void evaluate(const std::vector<ChannelBatchBuffer> &input_buffer) = 0;

    size_t numInputChannels() const {
        return base_.InputChannels;
    }

    size_t numOutputChannels() const {
        return base_.OutputChannels;
    }

    std::vector<ChannelBatchBuffer> *outputBuffer() {
        return &output_buffer_;
    }

protected:
    cmsStage base_;
    std::vector<ChannelBatchBuffer> output_buffer_;
};

class EvaluateCurvesFasterStage : public FasterStage {
public:
    explicit EvaluateCurvesFasterStage(const cmsStage &base) : FasterStage(base) {}

    void evaluate(const std::vector<ChannelBatchBuffer> &input_buffer) override {
        _cmsStageToneCurvesData *curvesData = reinterpret_cast<_cmsStageToneCurvesData *>(base_.Data);

        if (curvesData == NULL) return;
        if (curvesData->TheCurves == NULL) return;

        for (size_t i = 0; i < curvesData->nCurves; i++) {
            cmsToneCurve *const curve = curvesData->TheCurves[i];
            _cmsAssert(curve != NULL);
            const ChannelBatchBuffer &in = input_buffer[i];
            ChannelBatchBuffer &out = output_buffer_[i];

            if (curve->nSegments == 0) {
                batchEvaluateLimitedPrecision(curve, in, &out);
            } else {
                for (size_t batchIdx = 0; batchIdx < kFasterStageBatchSize; ++batchIdx) {
                    out[batchIdx] = cmsEvalToneCurveFloat(curve, in[batchIdx]);
                }
            }
        }
    }

private:
    static __m256 avxSaturate(register __m256 input) {
        input = _mm256_set1_ps(65535.0) * input + _mm256_set1_ps(0.5);
        return _mm256_floor_ps(
                _mm256_min_ps(_mm256_set1_ps(665535.0),
                              _mm256_max_ps(_mm256_set1_ps(0.0), input)));
    }

    static __m256 avxUnpack(register __m256i epi32_input) {
        return _mm256_mul_ps( _mm256_set1_ps(1.0f / 65535.0f), _mm256_cvtepi32_ps(epi32_input));
    }

    void batchEvaluateLimitedPrecision(
            const cmsToneCurve *curve, const ChannelBatchBuffer in,
            ChannelBatchBuffer *out) const {
        cmsUInt16Number inBuffer[kFasterStageBatchSize];
        cmsUInt16Number outBuffer[kFasterStageBatchSize];

        // Convert everything to uint16 using AVX.
        for (size_t batchIdx = 0; batchIdx < kFasterStageBatchSize; batchIdx += 16) {
            __m256 input1 = avxSaturate(_mm256_loadu_ps(&in[batchIdx]));
            __m256 input2 = avxSaturate(_mm256_loadu_ps(&in[batchIdx + 8]));
            __m256i packed = _mm256_packus_epi32(
                    _mm256_cvtps_epi32(input1),
                    _mm256_cvtps_epi32(input2));
            _mm256_storeu_si256(
                    reinterpret_cast<__m256i *>(&inBuffer[batchIdx]),
                    packed);
        }

        _cmsInterpFn16 curveFcn = curve->InterpParams->Interpolation.Lerp16;
        for (size_t batchIdx = 0; batchIdx < kFasterStageBatchSize; ++batchIdx) {
            // Check for 16 bits table. If so, this is a limited-precision tone curve
            curveFcn(&inBuffer[batchIdx], &outBuffer[batchIdx], curve->InterpParams);
        }

        // Convert back to float using AVX.
        for (size_t batchIdx = 0; batchIdx < kFasterStageBatchSize; batchIdx += 16) {
            __m256i packed = _mm256_loadu_si256(reinterpret_cast<__m256i *>(&outBuffer[batchIdx]));
            __m256 output1 = avxUnpack(_mm256_unpacklo_epi16(packed, _mm256_setzero_si256()));
            __m256 output2 = avxUnpack(_mm256_unpackhi_epi16(packed, _mm256_setzero_si256()));
            _mm256_storeu_ps(&(*out)[batchIdx], output1);
            _mm256_storeu_ps(&(*out)[batchIdx + 8], output2);
        }
    }
};

class EvaluateLab2XyzFasterStage : public FasterStage {
public:
    explicit EvaluateLab2XyzFasterStage(const cmsStage &base) : FasterStage(base) {}

    void evaluate(const std::vector<ChannelBatchBuffer> &input_buffer) override {
        const cmsCIEXYZ* WhitePoint = cmsD50_XYZ();
        for (size_t batchIdx = 0; batchIdx < kFasterStageBatchSize; ++batchIdx) {
            cmsCIELab Lab;
            cmsCIEXYZ XYZ;
            const cmsFloat64Number XYZadj = MAX_ENCODEABLE_XYZ;

            // V4 rules
            Lab.L = input_buffer[0][batchIdx] * 100.0;
            Lab.a = input_buffer[1][batchIdx] * 255.0 - 128.0;
            Lab.b = input_buffer[2][batchIdx] * 255.0 - 128.0;

            cmsFloat64Number x, y, z;

            y = (Lab.L + 16.0) / 116.0;
            x = y + 0.002 * Lab.a;
            z = y - 0.005 * Lab.b;

            XYZ.X = f_1(x) * WhitePoint->X;
            XYZ.Y = f_1(y) * WhitePoint->Y;
            XYZ.Z = f_1(z) * WhitePoint->Z;

            // From XYZ, range 0..19997 to 0..1.0, note that 1.99997 comes from 0xffff
            // encoded as 1.15 fixed point, so 1 + (32767.0 / 32768.0)
            output_buffer_[0][batchIdx] = (cmsFloat32Number) ((cmsFloat64Number) XYZ.X / XYZadj);
            output_buffer_[1][batchIdx] = (cmsFloat32Number) ((cmsFloat64Number) XYZ.Y / XYZadj);
            output_buffer_[2][batchIdx] = (cmsFloat32Number) ((cmsFloat64Number) XYZ.Z / XYZadj);
        }
    }

private:
    static cmsFloat64Number f_1(cmsFloat64Number t) {
        const cmsFloat64Number Limit = (24.0/116.0);

        if (t <= Limit) {
            return (108.0/841.0) * (t - (16.0/116.0));
        }

        return t * t * t;
    }
};

class EvaluateMatrixFasterStage : public FasterStage {
public:
    explicit EvaluateMatrixFasterStage(const cmsStage &base) : FasterStage(base) {}
    void evaluate(const std::vector<ChannelBatchBuffer> &input_buffer) override {
        cmsUInt32Number i, j;
        auto matrixData = reinterpret_cast<_cmsStageMatrixData*>(base_.Data);
        cmsFloat64Number Tmp;

        // Input is already in 0..1.0 notation
        for (size_t batchIdx = 0; batchIdx < kFasterStageBatchSize; ++batchIdx) {
            for (i = 0; i < base_.OutputChannels; i++) {

                Tmp = 0;
                for (j = 0; j < base_.InputChannels; j++) {
                    Tmp += input_buffer[j][batchIdx] * matrixData->Double[i * base_.InputChannels + j];
                }

                if (matrixData->Offset != NULL)
                    Tmp += matrixData->Offset[i];

                output_buffer_[i][batchIdx] = (cmsFloat32Number) Tmp;
            }
        }
        // Output in 0..1.0 domain
    }
};

/** Returns true if a faster pipeline could be assembled, false otherwise. */
bool batchPipeline(
        const _cmsOPTeval16Fn Eval16Fn,
        void *data,
        std::vector<std::unique_ptr<FasterStage>> *output) {
    if (Eval16Fn == _LUTeval16) {
        auto lut = reinterpret_cast<cmsPipeline*>(data);
        std::vector<std::unique_ptr<FasterStage>> tempFasterPipeline;
        for (cmsStage *mpe = lut->Elements; mpe != NULL; mpe = mpe->Next) {
            if (mpe->EvalPtr == &EvaluateLab2XYZ) {
                tempFasterPipeline.push_back(std::unique_ptr<FasterStage>(
                        new EvaluateLab2XyzFasterStage(*mpe)
                ));
            } else if (mpe->EvalPtr == &EvaluateMatrix) {
                tempFasterPipeline.push_back(std::unique_ptr<FasterStage>(
                        new EvaluateMatrixFasterStage(*mpe)
                ));
            } else if (mpe->EvalPtr == &EvaluateCurves) {
                tempFasterPipeline.push_back(std::unique_ptr<FasterStage>(
                        new EvaluateCurvesFasterStage(*mpe)
                ));
            } else {
                std::cout << "Failed to compile pipeline for stage "
                          << reinterpret_cast<uintptr_t>(mpe) << "\n";
                return false;
            }
        }
        *output = std::move(tempFasterPipeline);
        return true;
    } else {
        return false;
    }
}

void runFasterPipeline(
        const std::vector<std::unique_ptr<FasterStage>> &fasterPipeline
) {
    //...
}

// No gamut check, no cache, 16 bits
static
void PrecalculatedXFORM(_cmsTRANSFORM* p,
                        const void* in,
                        void* out, 
                        cmsUInt32Number PixelsPerLine,
                        cmsUInt32Number LineCount,
                        const cmsStride* Stride)
{
    register cmsUInt8Number* accum;
    register cmsUInt8Number* output;
    cmsUInt16Number wIn[cmsMAXCHANNELS], wOut[cmsMAXCHANNELS];
    cmsUInt32Number i, strideIn, strideOut;

    _cmsHandleExtraChannels(p, in, out, PixelsPerLine, LineCount, Stride);

    strideIn = 0;
    strideOut = 0;
    memset(wIn, 0, sizeof(wIn));
    memset(wOut, 0, sizeof(wOut));

    // Check if we can use the faster routine.
    std::vector<std::unique_ptr<FasterStage>> fasterPipeline;
    bool hasFasterPipeline = batchPipeline(p->Lut->Eval16Fn, p->Lut->Data, &fasterPipeline);
    if (hasFasterPipeline && PixelsPerLine >= 32) {
        const size_t numInputChannels = fasterPipeline[0]->numInputChannels();
        const size_t numOutputChannels = fasterPipeline[0]->numOutputChannels();
        std::vector<ChannelBatchBuffer> input_batch(numInputChannels);


        for (i = 0; i < LineCount; i++) {
            accum = (cmsUInt8Number *) in + strideIn;
            output = (cmsUInt8Number *) out + strideOut;

            for (size_t jBatch = 0; jBatch < PixelsPerLine; jBatch += kFasterStageBatchSize) {
                size_t numValid = std::min(PixelsPerLine - jBatch, kFasterStageBatchSize);

                // Get input. TODO: Vectorize
                for (int j = 0; j < numValid; ++j) {
                    accum = p->FromInput(p, wIn, accum, Stride->BytesPerPlaneIn);
                    for (int channel = 0; channel < numInputChannels; ++channel) {
                        input_batch[channel][j] = static_cast<float>(wIn[channel]) / 65535.0f;
                    }
                }

                // Run the pipeline
                auto currStage = &input_batch;
                for (auto &stage : fasterPipeline) {
                    stage->evaluate(*currStage);
                    currStage = stage->outputBuffer();
                }

                // Save the output. TODO: Vectorize
                for (int j = 0; j < numValid; ++j) {
                    for (int channel = 0; channel < numOutputChannels; ++channel) {
                        wOut[channel] = _cmsQuickSaturateWord((*currStage)[channel][j] * 65535.0);
                    }
                    output = p->ToOutput(p, wOut, output, Stride->BytesPerPlaneOut);
                }
            }

            strideIn += Stride->BytesPerLineIn;
            strideOut += Stride->BytesPerLineOut;
        }
    } else {
        for (i = 0; i < LineCount; i++) {

            accum = (cmsUInt8Number *) in + strideIn;
            output = (cmsUInt8Number *) out + strideOut;

            for (cmsUInt32Number j = 0; j < PixelsPerLine; j++) {
                accum = p->FromInput(p, wIn, accum, Stride->BytesPerPlaneIn);
                p->Lut->Eval16Fn(wIn, wOut, p->Lut->Data);
                output = p->ToOutput(p, wOut, output, Stride->BytesPerPlaneOut);
            }

            strideIn += Stride->BytesPerLineIn;
            strideOut += Stride->BytesPerLineOut;
        }
    }
}


// Auxiliary: Handle precalculated gamut check. The retrieval of context may be alittle bit slow, but this function is not critical.
static
void TransformOnePixelWithGamutCheck(_cmsTRANSFORM* p,
                                     const cmsUInt16Number wIn[],
                                     cmsUInt16Number wOut[])
{
    cmsUInt16Number wOutOfGamut;

    p ->GamutCheck ->Eval16Fn(wIn, &wOutOfGamut, p ->GamutCheck ->Data);
    if (wOutOfGamut >= 1) {

        cmsUInt16Number i;
        _cmsAlarmCodesChunkType* ContextAlarmCodes = (_cmsAlarmCodesChunkType*) _cmsContextGetClientChunk(p->ContextID, AlarmCodesContext);        

        for (i=0; i < p ->Lut->OutputChannels; i++) {

            wOut[i] = ContextAlarmCodes ->AlarmCodes[i];
        }
    }
    else
        p ->Lut ->Eval16Fn(wIn, wOut, p -> Lut->Data);
}

// Gamut check, No cache, 16 bits.
static
void PrecalculatedXFORMGamutCheck(_cmsTRANSFORM* p,
                                  const void* in,
                                  void* out, 
                                  cmsUInt32Number PixelsPerLine,
                                  cmsUInt32Number LineCount,
                                  const cmsStride* Stride)
{
    cmsUInt8Number* accum;
    cmsUInt8Number* output;
    cmsUInt16Number wIn[cmsMAXCHANNELS], wOut[cmsMAXCHANNELS];
    cmsUInt32Number i, j, strideIn, strideOut;

    _cmsHandleExtraChannels(p, in, out, PixelsPerLine, LineCount, Stride);

    strideIn = 0;
    strideOut = 0;
    memset(wIn, 0, sizeof(wIn));
    memset(wOut, 0, sizeof(wOut));

    for (i = 0; i < LineCount; i++) {

           accum = (cmsUInt8Number*)in + strideIn;
           output = (cmsUInt8Number*)out + strideOut;

           for (j = 0; j < PixelsPerLine; j++) {

                  accum = p->FromInput(p, wIn, accum, Stride->BytesPerPlaneIn);
                  TransformOnePixelWithGamutCheck(p, wIn, wOut);
                  output = p->ToOutput(p, wOut, output, Stride->BytesPerPlaneOut);
           }

           strideIn += Stride->BytesPerLineIn;
           strideOut += Stride->BytesPerLineOut;
    }
}


// No gamut check, Cach�, 16 bits,
static
void CachedXFORM(_cmsTRANSFORM* p,
                 const void* in,
                 void* out,
                 cmsUInt32Number PixelsPerLine,
                 cmsUInt32Number LineCount,
                 const cmsStride* Stride)
{
    cmsUInt8Number* accum;
    cmsUInt8Number* output;
    cmsUInt16Number wIn[cmsMAXCHANNELS], wOut[cmsMAXCHANNELS];
    _cmsCACHE Cache;
    cmsUInt32Number i, j, strideIn, strideOut;

    _cmsHandleExtraChannels(p, in, out, PixelsPerLine, LineCount, Stride);

    // Empty buffers for quick memcmp
    memset(wIn, 0, sizeof(wIn));
    memset(wOut, 0, sizeof(wOut));

    // Get copy of zero cache
    memcpy(&Cache, &p->Cache, sizeof(Cache));

    strideIn = 0;
    strideOut = 0;

    for (i = 0; i < LineCount; i++) {

        accum = (cmsUInt8Number*)in + strideIn;
        output = (cmsUInt8Number*)out + strideOut;

        for (j = 0; j < PixelsPerLine; j++) {

            accum = p->FromInput(p, wIn, accum, Stride->BytesPerPlaneIn);

            if (memcmp(wIn, Cache.CacheIn, sizeof(Cache.CacheIn)) == 0) {

                memcpy(wOut, Cache.CacheOut, sizeof(Cache.CacheOut));
            }
            else {
                p->Lut->Eval16Fn(wIn, wOut, p->Lut->Data);

                memcpy(Cache.CacheIn, wIn, sizeof(Cache.CacheIn));
                memcpy(Cache.CacheOut, wOut, sizeof(Cache.CacheOut));
            }

            output = p->ToOutput(p, wOut, output, Stride->BytesPerPlaneOut);
        }

        strideIn += Stride->BytesPerLineIn;
        strideOut += Stride->BytesPerLineOut;
    }
}

// All those nice features together
static
void CachedXFORMGamutCheck(_cmsTRANSFORM* p,
                           const void* in,
                           void* out, 
                           cmsUInt32Number PixelsPerLine,
                           cmsUInt32Number LineCount,
                           const cmsStride* Stride)
{
    cmsUInt8Number* accum;
    cmsUInt8Number* output;
    cmsUInt16Number wIn[cmsMAXCHANNELS], wOut[cmsMAXCHANNELS];
    _cmsCACHE Cache;
    cmsUInt32Number i, j, strideIn, strideOut;

    _cmsHandleExtraChannels(p, in, out, PixelsPerLine, LineCount, Stride);

    // Empty buffers for quick memcmp
    memset(wIn, 0, sizeof(wIn));
    memset(wOut, 0, sizeof(wOut));

    // Get copy of zero cache
    memcpy(&Cache, &p->Cache, sizeof(Cache));

    strideIn = 0;
    strideOut = 0;

    for (i = 0; i < LineCount; i++) {

        accum = (cmsUInt8Number*)in + strideIn;
        output = (cmsUInt8Number*)out + strideOut;

        for (j = 0; j < PixelsPerLine; j++) {

            accum = p->FromInput(p, wIn, accum, Stride->BytesPerPlaneIn);

            if (memcmp(wIn, Cache.CacheIn, sizeof(Cache.CacheIn)) == 0) {

                memcpy(wOut, Cache.CacheOut, sizeof(Cache.CacheOut));
            }
            else {
                TransformOnePixelWithGamutCheck(p, wIn, wOut);

                memcpy(Cache.CacheIn, wIn, sizeof(Cache.CacheIn));
                memcpy(Cache.CacheOut, wOut, sizeof(Cache.CacheOut));
            }

            output = p->ToOutput(p, wOut, output, Stride->BytesPerPlaneOut);
        }

        strideIn += Stride->BytesPerLineIn;
        strideOut += Stride->BytesPerLineOut;
    }
}

// Transform plug-ins ----------------------------------------------------------------------------------------------------

// List of used-defined transform factories
typedef struct _cmsTransformCollection_st {

    _cmsTransform2Factory  Factory;
    cmsBool                OldXform;   // Factory returns xform function in the old style

    struct _cmsTransformCollection_st *Next;

} _cmsTransformCollection;

// The linked list head
_cmsTransformPluginChunkType _cmsTransformPluginChunk = { NULL };


// Duplicates the zone of memory used by the plug-in in the new context
static
void DupPluginTransformList(struct _cmsContext_struct* ctx, 
                                               const struct _cmsContext_struct* src)
{
   _cmsTransformPluginChunkType newHead = { NULL };
   _cmsTransformCollection*  entry;
   _cmsTransformCollection*  Anterior = NULL;
   _cmsTransformPluginChunkType* head = (_cmsTransformPluginChunkType*) src->chunks[TransformPlugin];

    // Walk the list copying all nodes
   for (entry = head->TransformCollection;
        entry != NULL;
        entry = entry ->Next) {

            _cmsTransformCollection *newEntry = ( _cmsTransformCollection *) _cmsSubAllocDup(ctx ->MemPool, entry, sizeof(_cmsTransformCollection));
   
            if (newEntry == NULL) 
                return;

            // We want to keep the linked list order, so this is a little bit tricky
            newEntry -> Next = NULL;
            if (Anterior)
                Anterior -> Next = newEntry;
     
            Anterior = newEntry;

            if (newHead.TransformCollection == NULL)
                newHead.TransformCollection = newEntry;
    }

  ctx ->chunks[TransformPlugin] = _cmsSubAllocDup(ctx->MemPool, &newHead, sizeof(_cmsTransformPluginChunkType));
}

// Allocates memory for transform plugin factory
void _cmsAllocTransformPluginChunk(struct _cmsContext_struct* ctx, 
                                        const struct _cmsContext_struct* src)
{
    if (src != NULL) {

        // Copy all linked list
        DupPluginTransformList(ctx, src);
    }
    else {
        static _cmsTransformPluginChunkType TransformPluginChunkType = { NULL };
        ctx ->chunks[TransformPlugin] = _cmsSubAllocDup(ctx ->MemPool, &TransformPluginChunkType, sizeof(_cmsTransformPluginChunkType));
    }
}

// Adaptor for old versions of plug-in
static
void _cmsTransform2toTransformAdaptor(struct _cmstransform_struct *CMMcargo,
                                      const void* InputBuffer,
                                      void* OutputBuffer,
                                      cmsUInt32Number PixelsPerLine,
                                      cmsUInt32Number LineCount,
                                      const cmsStride* Stride)
{
     
       cmsUInt32Number i, strideIn, strideOut;

       _cmsHandleExtraChannels(CMMcargo, InputBuffer, OutputBuffer, PixelsPerLine, LineCount, Stride);

       strideIn = 0;
       strideOut = 0;

       for (i = 0; i < LineCount; i++) {

              void *accum = (cmsUInt8Number*)InputBuffer + strideIn;
              void *output = (cmsUInt8Number*)OutputBuffer + strideOut;

              CMMcargo->OldXform(CMMcargo, accum, output, PixelsPerLine, Stride->BytesPerPlaneIn);

              strideIn += Stride->BytesPerLineIn;
              strideOut += Stride->BytesPerLineOut;
       }
}



// Register new ways to transform
cmsBool  _cmsRegisterTransformPlugin(cmsContext ContextID, cmsPluginBase* Data)
{
    cmsPluginTransform* Plugin = (cmsPluginTransform*) Data;
    _cmsTransformCollection* fl;
    _cmsTransformPluginChunkType* ctx = ( _cmsTransformPluginChunkType*) _cmsContextGetClientChunk(ContextID,TransformPlugin);

    if (Data == NULL) {

        // Free the chain. Memory is safely freed at exit
        ctx->TransformCollection = NULL;
        return TRUE;
    }

    // Factory callback is required
    if (Plugin->factories.xform == NULL) return FALSE;


    fl = (_cmsTransformCollection*) _cmsPluginMalloc(ContextID, sizeof(_cmsTransformCollection));
    if (fl == NULL) return FALSE;

    // Check for full xform plug-ins previous to 2.8, we would need an adapter in that case
    if (Plugin->base.ExpectedVersion < 2080) {

           fl->OldXform = TRUE;
    }
    else
           fl->OldXform = FALSE;

    // Copy the parameters
    fl->Factory = Plugin->factories.xform;

    // Keep linked list
    fl ->Next = ctx->TransformCollection;
    ctx->TransformCollection = fl;

    // All is ok
    return TRUE;
}


void CMSEXPORT _cmsSetTransformUserData(struct _cmstransform_struct *CMMcargo, void* ptr, _cmsFreeUserDataFn FreePrivateDataFn)
{
    _cmsAssert(CMMcargo != NULL);
    CMMcargo ->UserData = ptr;
    CMMcargo ->FreeUserData = FreePrivateDataFn;
}

// returns the pointer defined by the plug-in to store private data
void * CMSEXPORT _cmsGetTransformUserData(struct _cmstransform_struct *CMMcargo)
{
    _cmsAssert(CMMcargo != NULL);
    return CMMcargo ->UserData;
}

// returns the current formatters
void CMSEXPORT _cmsGetTransformFormatters16(struct _cmstransform_struct *CMMcargo, cmsFormatter16* FromInput, cmsFormatter16* ToOutput)
{
     _cmsAssert(CMMcargo != NULL);
     if (FromInput) *FromInput = CMMcargo ->FromInput;
     if (ToOutput)  *ToOutput  = CMMcargo ->ToOutput;
}

void CMSEXPORT _cmsGetTransformFormattersFloat(struct _cmstransform_struct *CMMcargo, cmsFormatterFloat* FromInput, cmsFormatterFloat* ToOutput)
{
     _cmsAssert(CMMcargo != NULL);
     if (FromInput) *FromInput = CMMcargo ->FromInputFloat;
     if (ToOutput)  *ToOutput  = CMMcargo ->ToOutputFloat;
}


// Allocate transform struct and set it to defaults. Ask the optimization plug-in about if those formats are proper
// for separated transforms. If this is the case,
static
_cmsTRANSFORM* AllocEmptyTransform(cmsContext ContextID, cmsPipeline* lut,
                                               cmsUInt32Number Intent, cmsUInt32Number* InputFormat, cmsUInt32Number* OutputFormat, cmsUInt32Number* dwFlags)
{
     _cmsTransformPluginChunkType* ctx = ( _cmsTransformPluginChunkType*) _cmsContextGetClientChunk(ContextID, TransformPlugin);
     _cmsTransformCollection* Plugin;

       // Allocate needed memory
       _cmsTRANSFORM* p = (_cmsTRANSFORM*)_cmsMallocZero(ContextID, sizeof(_cmsTRANSFORM));
       if (!p) {
              cmsPipelineFree(lut);
              return NULL;
       }

       // Store the proposed pipeline
       p->Lut = lut;

       // Let's see if any plug-in want to do the transform by itself
       if (p->Lut != NULL) {

              for (Plugin = ctx->TransformCollection;
                     Plugin != NULL;
                     Plugin = Plugin->Next) {

                     if (Plugin->Factory(&p->xform, &p->UserData, &p->FreeUserData, &p->Lut, InputFormat, OutputFormat, dwFlags)) {

                            // Last plugin in the declaration order takes control. We just keep
                            // the original parameters as a logging. 
                            // Note that cmsFLAGS_CAN_CHANGE_FORMATTER is not set, so by default 
                            // an optimized transform is not reusable. The plug-in can, however, change
                            // the flags and make it suitable.

                            p->ContextID = ContextID;
                            p->InputFormat = *InputFormat;
                            p->OutputFormat = *OutputFormat;
                            p->dwOriginalFlags = *dwFlags;

                            // Fill the formatters just in case the optimized routine is interested.
                            // No error is thrown if the formatter doesn't exist. It is up to the optimization 
                            // factory to decide what to do in those cases.
                            p->FromInput = _cmsGetFormatter(ContextID, *InputFormat, cmsFormatterInput, CMS_PACK_FLAGS_16BITS).Fmt16;
                            p->ToOutput = _cmsGetFormatter(ContextID, *OutputFormat, cmsFormatterOutput, CMS_PACK_FLAGS_16BITS).Fmt16;
                            p->FromInputFloat = _cmsGetFormatter(ContextID, *InputFormat, cmsFormatterInput, CMS_PACK_FLAGS_FLOAT).FmtFloat;
                            p->ToOutputFloat = _cmsGetFormatter(ContextID, *OutputFormat, cmsFormatterOutput, CMS_PACK_FLAGS_FLOAT).FmtFloat;

                            // Save the day? (Ignore the warning)
                            if (Plugin->OldXform) {
                                   p->OldXform = (_cmsTransformFn) p->xform;
                                   p->xform = _cmsTransform2toTransformAdaptor;
                            }
                             
                            return p;
                     }
              }

              // Not suitable for the transform plug-in, let's check  the pipeline plug-in
              _cmsOptimizePipeline(ContextID, &p->Lut, Intent, InputFormat, OutputFormat, dwFlags);
       }

    // Check whatever this is a true floating point transform
    if (_cmsFormatterIsFloat(*InputFormat) && _cmsFormatterIsFloat(*OutputFormat)) {

        // Get formatter function always return a valid union, but the contents of this union may be NULL.
        p ->FromInputFloat = _cmsGetFormatter(ContextID, *InputFormat,  cmsFormatterInput, CMS_PACK_FLAGS_FLOAT).FmtFloat;
        p ->ToOutputFloat  = _cmsGetFormatter(ContextID, *OutputFormat, cmsFormatterOutput, CMS_PACK_FLAGS_FLOAT).FmtFloat;
        *dwFlags |= cmsFLAGS_CAN_CHANGE_FORMATTER;

        if (p ->FromInputFloat == NULL || p ->ToOutputFloat == NULL) {

            cmsSignalError(ContextID, cmsERROR_UNKNOWN_EXTENSION, "Unsupported raster format");
            cmsDeleteTransform(p);
            return NULL;
        }

        if (*dwFlags & cmsFLAGS_NULLTRANSFORM) {

            p ->xform = NullFloatXFORM;
        }
        else {
            // Float transforms don't use cache, always are non-NULL
            p ->xform = FloatXFORM;
        }

    }
    else {

        if (*InputFormat == 0 && *OutputFormat == 0) {
            p ->FromInput = p ->ToOutput = NULL;
            *dwFlags |= cmsFLAGS_CAN_CHANGE_FORMATTER;
        }
        else {

            cmsUInt32Number BytesPerPixelInput;

            p ->FromInput = _cmsGetFormatter(ContextID, *InputFormat,  cmsFormatterInput, CMS_PACK_FLAGS_16BITS).Fmt16;
            p ->ToOutput  = _cmsGetFormatter(ContextID, *OutputFormat, cmsFormatterOutput, CMS_PACK_FLAGS_16BITS).Fmt16;

            if (p ->FromInput == NULL || p ->ToOutput == NULL) {

                cmsSignalError(ContextID, cmsERROR_UNKNOWN_EXTENSION, "Unsupported raster format");
                cmsDeleteTransform(p);
                return NULL;
            }

            BytesPerPixelInput = T_BYTES(p ->InputFormat);
            if (BytesPerPixelInput == 0 || BytesPerPixelInput >= 2)
                   *dwFlags |= cmsFLAGS_CAN_CHANGE_FORMATTER;

        }

        if (*dwFlags & cmsFLAGS_NULLTRANSFORM) {

            p ->xform = NullXFORM;
        }
        else {
            if (*dwFlags & cmsFLAGS_NOCACHE) {

                if (*dwFlags & cmsFLAGS_GAMUTCHECK)
                    p ->xform = PrecalculatedXFORMGamutCheck;  // Gamut check, no cache
                else
                    p ->xform = PrecalculatedXFORM;  // No cache, no gamut check
            }
            else {

                if (*dwFlags & cmsFLAGS_GAMUTCHECK)
                    p ->xform = CachedXFORMGamutCheck;    // Gamut check, cache
                else
                    p ->xform = CachedXFORM;  // No gamut check, cache

            }
        }
    }

    p ->InputFormat     = *InputFormat;
    p ->OutputFormat    = *OutputFormat;
    p ->dwOriginalFlags = *dwFlags;
    p ->ContextID       = ContextID;
    p ->UserData        = NULL;
    return p;
}

static
cmsBool GetXFormColorSpaces(cmsUInt32Number nProfiles, cmsHPROFILE hProfiles[], cmsColorSpaceSignature* Input, cmsColorSpaceSignature* Output)
{
    cmsColorSpaceSignature ColorSpaceIn, ColorSpaceOut;
    cmsColorSpaceSignature PostColorSpace;
    cmsUInt32Number i;

    if (nProfiles == 0) return FALSE;
    if (hProfiles[0] == NULL) return FALSE;

    *Input = PostColorSpace = cmsGetColorSpace(hProfiles[0]);

    for (i=0; i < nProfiles; i++) {

        cmsProfileClassSignature cls;
        cmsHPROFILE hProfile = hProfiles[i];

        int lIsInput = (PostColorSpace != cmsSigXYZData) &&
                       (PostColorSpace != cmsSigLabData);

        if (hProfile == NULL) return FALSE;

        cls = cmsGetDeviceClass(hProfile);

        if (cls == cmsSigNamedColorClass) {

            ColorSpaceIn    = cmsSig1colorData;
            ColorSpaceOut   = (nProfiles > 1) ? cmsGetPCS(hProfile) : cmsGetColorSpace(hProfile);
        }
        else
        if (lIsInput || (cls == cmsSigLinkClass)) {

            ColorSpaceIn    = cmsGetColorSpace(hProfile);
            ColorSpaceOut   = cmsGetPCS(hProfile);
        }
        else
        {
            ColorSpaceIn    = cmsGetPCS(hProfile);
            ColorSpaceOut   = cmsGetColorSpace(hProfile);
        }

        if (i==0)
            *Input = ColorSpaceIn;

        PostColorSpace = ColorSpaceOut;
    }

    *Output = PostColorSpace;

    return TRUE;
}

// Check colorspace
static
cmsBool  IsProperColorSpace(cmsColorSpaceSignature Check, cmsUInt32Number dwFormat)
{
    int Space1 = (int) T_COLORSPACE(dwFormat);
    int Space2 = _cmsLCMScolorSpace(Check);

    if (Space1 == PT_ANY) return TRUE;
    if (Space1 == Space2) return TRUE;

    if (Space1 == PT_LabV2 && Space2 == PT_Lab) return TRUE;
    if (Space1 == PT_Lab   && Space2 == PT_LabV2) return TRUE;

    return FALSE;
}

// ----------------------------------------------------------------------------------------------------------------

// Jun-21-2000: Some profiles (those that comes with W2K) comes
// with the media white (media black?) x 100. Add a sanity check

static
void NormalizeXYZ(cmsCIEXYZ* Dest)
{
    while (Dest -> X > 2. &&
           Dest -> Y > 2. &&
           Dest -> Z > 2.) {

               Dest -> X /= 10.;
               Dest -> Y /= 10.;
               Dest -> Z /= 10.;
       }
}

static
void SetWhitePoint(cmsCIEXYZ* wtPt, const cmsCIEXYZ* src)
{
    if (src == NULL) {
        wtPt ->X = cmsD50X;
        wtPt ->Y = cmsD50Y;
        wtPt ->Z = cmsD50Z;
    }
    else {
        wtPt ->X = src->X;
        wtPt ->Y = src->Y;
        wtPt ->Z = src->Z;

        NormalizeXYZ(wtPt);
    }

}

// New to lcms 2.0 -- have all parameters available.
cmsHTRANSFORM CMSEXPORT cmsCreateExtendedTransform(cmsContext ContextID,
                                                   cmsUInt32Number nProfiles, cmsHPROFILE hProfiles[],
                                                   cmsBool  BPC[],
                                                   cmsUInt32Number Intents[],
                                                   cmsFloat64Number AdaptationStates[],
                                                   cmsHPROFILE hGamutProfile,
                                                   cmsUInt32Number nGamutPCSposition,
                                                   cmsUInt32Number InputFormat,
                                                   cmsUInt32Number OutputFormat,
                                                   cmsUInt32Number dwFlags)
{
    _cmsTRANSFORM* xform;    
    cmsColorSpaceSignature EntryColorSpace;
    cmsColorSpaceSignature ExitColorSpace;
    cmsPipeline* Lut;
    cmsUInt32Number LastIntent = Intents[nProfiles-1];

    // If it is a fake transform
    if (dwFlags & cmsFLAGS_NULLTRANSFORM)
    {
        return AllocEmptyTransform(ContextID, NULL, INTENT_PERCEPTUAL, &InputFormat, &OutputFormat, &dwFlags);
    }

    // If gamut check is requested, make sure we have a gamut profile
    if (dwFlags & cmsFLAGS_GAMUTCHECK) {
        if (hGamutProfile == NULL) dwFlags &= ~cmsFLAGS_GAMUTCHECK;
    }

    // On floating point transforms, inhibit cache
    if (_cmsFormatterIsFloat(InputFormat) || _cmsFormatterIsFloat(OutputFormat))
        dwFlags |= cmsFLAGS_NOCACHE;

    // Mark entry/exit spaces
    if (!GetXFormColorSpaces(nProfiles, hProfiles, &EntryColorSpace, &ExitColorSpace)) {
        cmsSignalError(ContextID, cmsERROR_NULL, "NULL input profiles on transform");
        return NULL;
    }

    // Check if proper colorspaces
    if (!IsProperColorSpace(EntryColorSpace, InputFormat)) {
        cmsSignalError(ContextID, cmsERROR_COLORSPACE_CHECK, "Wrong input color space on transform");
        return NULL;
    }

    if (!IsProperColorSpace(ExitColorSpace, OutputFormat)) {
        cmsSignalError(ContextID, cmsERROR_COLORSPACE_CHECK, "Wrong output color space on transform");
        return NULL;
    }

    // Create a pipeline with all transformations
    Lut = _cmsLinkProfiles(ContextID, nProfiles, Intents, hProfiles, BPC, AdaptationStates, dwFlags);
    if (Lut == NULL) {
        cmsSignalError(ContextID, cmsERROR_NOT_SUITABLE, "Couldn't link the profiles");
        return NULL;
    }

    // Check channel count
    if ((cmsChannelsOf(EntryColorSpace) != cmsPipelineInputChannels(Lut)) ||
        (cmsChannelsOf(ExitColorSpace)  != cmsPipelineOutputChannels(Lut))) {
        cmsPipelineFree(Lut);
        cmsSignalError(ContextID, cmsERROR_NOT_SUITABLE, "Channel count doesn't match. Profile is corrupted");
        return NULL;
    }


    // All seems ok
    xform = AllocEmptyTransform(ContextID, Lut, LastIntent, &InputFormat, &OutputFormat, &dwFlags);
    if (xform == NULL) {
        return NULL;
    }

    // Keep values
    xform ->EntryColorSpace = EntryColorSpace;
    xform ->ExitColorSpace  = ExitColorSpace;
    xform ->RenderingIntent = Intents[nProfiles-1];

    // Take white points
    SetWhitePoint(&xform->EntryWhitePoint, (cmsCIEXYZ*) cmsReadTag(hProfiles[0], cmsSigMediaWhitePointTag));
    SetWhitePoint(&xform->ExitWhitePoint,  (cmsCIEXYZ*) cmsReadTag(hProfiles[nProfiles-1], cmsSigMediaWhitePointTag));
   

    // Create a gamut check LUT if requested
    if (hGamutProfile != NULL && (dwFlags & cmsFLAGS_GAMUTCHECK))
        xform ->GamutCheck  = _cmsCreateGamutCheckPipeline(ContextID, hProfiles,
                                                        BPC, Intents,
                                                        AdaptationStates,
                                                        nGamutPCSposition,
                                                        hGamutProfile);


    // Try to read input and output colorant table
    if (cmsIsTag(hProfiles[0], cmsSigColorantTableTag)) {

        // Input table can only come in this way.
        xform ->InputColorant = cmsDupNamedColorList((cmsNAMEDCOLORLIST*) cmsReadTag(hProfiles[0], cmsSigColorantTableTag));
    }

    // Output is a little bit more complex.
    if (cmsGetDeviceClass(hProfiles[nProfiles-1]) == cmsSigLinkClass) {

        // This tag may exist only on devicelink profiles.
        if (cmsIsTag(hProfiles[nProfiles-1], cmsSigColorantTableOutTag)) {

            // It may be NULL if error
            xform ->OutputColorant = cmsDupNamedColorList((cmsNAMEDCOLORLIST*) cmsReadTag(hProfiles[nProfiles-1], cmsSigColorantTableOutTag));
        }

    } else {

        if (cmsIsTag(hProfiles[nProfiles-1], cmsSigColorantTableTag)) {

            xform -> OutputColorant = cmsDupNamedColorList((cmsNAMEDCOLORLIST*) cmsReadTag(hProfiles[nProfiles-1], cmsSigColorantTableTag));
        }
    }

    // Store the sequence of profiles
    if (dwFlags & cmsFLAGS_KEEP_SEQUENCE) {
        xform ->Sequence = _cmsCompileProfileSequence(ContextID, nProfiles, hProfiles);
    }
    else
        xform ->Sequence = NULL;

    // If this is a cached transform, init first value, which is zero (16 bits only)
    if (!(dwFlags & cmsFLAGS_NOCACHE)) {

        memset(&xform ->Cache.CacheIn, 0, sizeof(xform ->Cache.CacheIn));

        if (xform ->GamutCheck != NULL) {
            TransformOnePixelWithGamutCheck(xform, xform ->Cache.CacheIn, xform->Cache.CacheOut);
        }
        else {

            xform ->Lut ->Eval16Fn(xform ->Cache.CacheIn, xform->Cache.CacheOut, xform -> Lut->Data);
        }

    }

    return (cmsHTRANSFORM) xform;
}

// Multiprofile transforms: Gamut check is not available here, as it is unclear from which profile the gamut comes.
cmsHTRANSFORM CMSEXPORT cmsCreateMultiprofileTransformTHR(cmsContext ContextID,
                                                       cmsHPROFILE hProfiles[],
                                                       cmsUInt32Number nProfiles,
                                                       cmsUInt32Number InputFormat,
                                                       cmsUInt32Number OutputFormat,
                                                       cmsUInt32Number Intent,
                                                       cmsUInt32Number dwFlags)
{
    cmsUInt32Number i;
    cmsBool BPC[256];
    cmsUInt32Number Intents[256];
    cmsFloat64Number AdaptationStates[256];

    if (nProfiles <= 0 || nProfiles > 255) {
         cmsSignalError(ContextID, cmsERROR_RANGE, "Wrong number of profiles. 1..255 expected, %d found.", nProfiles);
        return NULL;
    }

    for (i=0; i < nProfiles; i++) {
        BPC[i] = dwFlags & cmsFLAGS_BLACKPOINTCOMPENSATION ? TRUE : FALSE;
        Intents[i] = Intent;
        AdaptationStates[i] = cmsSetAdaptationStateTHR(ContextID, -1);
    }


    return cmsCreateExtendedTransform(ContextID, nProfiles, hProfiles, BPC, Intents, AdaptationStates, NULL, 0, InputFormat, OutputFormat, dwFlags);
}



cmsHTRANSFORM CMSEXPORT cmsCreateMultiprofileTransform(cmsHPROFILE hProfiles[],
                                                  cmsUInt32Number nProfiles,
                                                  cmsUInt32Number InputFormat,
                                                  cmsUInt32Number OutputFormat,
                                                  cmsUInt32Number Intent,
                                                  cmsUInt32Number dwFlags)
{

    if (nProfiles <= 0 || nProfiles > 255) {
         cmsSignalError(NULL, cmsERROR_RANGE, "Wrong number of profiles. 1..255 expected, %d found.", nProfiles);
         return NULL;
    }

    return cmsCreateMultiprofileTransformTHR(cmsGetProfileContextID(hProfiles[0]),
                                                  hProfiles,
                                                  nProfiles,
                                                  InputFormat,
                                                  OutputFormat,
                                                  Intent,
                                                  dwFlags);
}

cmsHTRANSFORM CMSEXPORT cmsCreateTransformTHR(cmsContext ContextID,
                                              cmsHPROFILE Input,
                                              cmsUInt32Number InputFormat,
                                              cmsHPROFILE Output,
                                              cmsUInt32Number OutputFormat,
                                              cmsUInt32Number Intent,
                                              cmsUInt32Number dwFlags)
{

    cmsHPROFILE hArray[2];

    hArray[0] = Input;
    hArray[1] = Output;

    return cmsCreateMultiprofileTransformTHR(ContextID, hArray, Output == NULL ? 1U : 2U, InputFormat, OutputFormat, Intent, dwFlags);
}

CMSAPI cmsHTRANSFORM CMSEXPORT cmsCreateTransform(cmsHPROFILE Input,
                                                  cmsUInt32Number InputFormat,
                                                  cmsHPROFILE Output,
                                                  cmsUInt32Number OutputFormat,
                                                  cmsUInt32Number Intent,
                                                  cmsUInt32Number dwFlags)
{
    return cmsCreateTransformTHR(cmsGetProfileContextID(Input), Input, InputFormat, Output, OutputFormat, Intent, dwFlags);
}


cmsHTRANSFORM CMSEXPORT cmsCreateProofingTransformTHR(cmsContext ContextID,
                                                   cmsHPROFILE InputProfile,
                                                   cmsUInt32Number InputFormat,
                                                   cmsHPROFILE OutputProfile,
                                                   cmsUInt32Number OutputFormat,
                                                   cmsHPROFILE ProofingProfile,
                                                   cmsUInt32Number nIntent,
                                                   cmsUInt32Number ProofingIntent,
                                                   cmsUInt32Number dwFlags)
{
    cmsHPROFILE hArray[4];
    cmsUInt32Number Intents[4];
    cmsBool  BPC[4];
    cmsFloat64Number Adaptation[4];
    cmsBool  DoBPC = (dwFlags & cmsFLAGS_BLACKPOINTCOMPENSATION) ? TRUE : FALSE;


    hArray[0]  = InputProfile; hArray[1] = ProofingProfile; hArray[2]  = ProofingProfile;               hArray[3] = OutputProfile;
    Intents[0] = nIntent;      Intents[1] = nIntent;        Intents[2] = INTENT_RELATIVE_COLORIMETRIC;  Intents[3] = ProofingIntent;
    BPC[0]     = DoBPC;        BPC[1] = DoBPC;              BPC[2] = 0;                                 BPC[3] = 0;

    Adaptation[0] = Adaptation[1] = Adaptation[2] = Adaptation[3] = cmsSetAdaptationStateTHR(ContextID, -1);

    if (!(dwFlags & (cmsFLAGS_SOFTPROOFING|cmsFLAGS_GAMUTCHECK)))
        return cmsCreateTransformTHR(ContextID, InputProfile, InputFormat, OutputProfile, OutputFormat, nIntent, dwFlags);

    return cmsCreateExtendedTransform(ContextID, 4, hArray, BPC, Intents, Adaptation,
                                        ProofingProfile, 1, InputFormat, OutputFormat, dwFlags);

}


cmsHTRANSFORM CMSEXPORT cmsCreateProofingTransform(cmsHPROFILE InputProfile,
                                                   cmsUInt32Number InputFormat,
                                                   cmsHPROFILE OutputProfile,
                                                   cmsUInt32Number OutputFormat,
                                                   cmsHPROFILE ProofingProfile,
                                                   cmsUInt32Number nIntent,
                                                   cmsUInt32Number ProofingIntent,
                                                   cmsUInt32Number dwFlags)
{
    return cmsCreateProofingTransformTHR(cmsGetProfileContextID(InputProfile),
                                                   InputProfile,
                                                   InputFormat,
                                                   OutputProfile,
                                                   OutputFormat,
                                                   ProofingProfile,
                                                   nIntent,
                                                   ProofingIntent,
                                                   dwFlags);
}


// Grab the ContextID from an open transform. Returns NULL if a NULL transform is passed
cmsContext CMSEXPORT cmsGetTransformContextID(cmsHTRANSFORM hTransform)
{
    _cmsTRANSFORM* xform = (_cmsTRANSFORM*) hTransform;

    if (xform == NULL) return NULL;
    return xform -> ContextID;
}

// Grab the input/output formats
cmsUInt32Number CMSEXPORT cmsGetTransformInputFormat(cmsHTRANSFORM hTransform)
{
    _cmsTRANSFORM* xform = (_cmsTRANSFORM*) hTransform;

    if (xform == NULL) return 0;
    return xform->InputFormat;
}

cmsUInt32Number CMSEXPORT cmsGetTransformOutputFormat(cmsHTRANSFORM hTransform)
{
    _cmsTRANSFORM* xform = (_cmsTRANSFORM*) hTransform;

    if (xform == NULL) return 0;
    return xform->OutputFormat;
}

// For backwards compatibility
cmsBool CMSEXPORT cmsChangeBuffersFormat(cmsHTRANSFORM hTransform,
                                         cmsUInt32Number InputFormat,
                                         cmsUInt32Number OutputFormat)
{
    _cmsTRANSFORM* xform = (_cmsTRANSFORM*) hTransform;
    cmsFormatter16 FromInput, ToOutput;


    // We only can afford to change formatters if previous transform is at least 16 bits
    if (!(xform ->dwOriginalFlags & cmsFLAGS_CAN_CHANGE_FORMATTER)) {

        cmsSignalError(xform ->ContextID, cmsERROR_NOT_SUITABLE, "cmsChangeBuffersFormat works only on transforms created originally with at least 16 bits of precision");
        return FALSE;
    }

    FromInput = _cmsGetFormatter(xform->ContextID, InputFormat,  cmsFormatterInput, CMS_PACK_FLAGS_16BITS).Fmt16;
    ToOutput  = _cmsGetFormatter(xform->ContextID, OutputFormat, cmsFormatterOutput, CMS_PACK_FLAGS_16BITS).Fmt16;

    if (FromInput == NULL || ToOutput == NULL) {

        cmsSignalError(xform -> ContextID, cmsERROR_UNKNOWN_EXTENSION, "Unsupported raster format");
        return FALSE;
    }

    xform ->InputFormat  = InputFormat;
    xform ->OutputFormat = OutputFormat;
    xform ->FromInput    = FromInput;
    xform ->ToOutput     = ToOutput;
    return TRUE;
}
