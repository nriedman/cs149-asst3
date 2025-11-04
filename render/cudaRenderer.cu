#include <string>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"

////////////////////////////////////////////////////////////////////////////////////////
// CUDA Circle Renderer 
////////////////////////////////////////////////////////////////////////////////////////

struct GlobalConstants {

    SceneName sceneName;

    int numCircles;
    float* position;
    float* velocity;
    float* color;
    float* radius;

    int imageWidth;
    int imageHeight;
    float* imageData;
    // Tile binning metadata (set prior to render())
    int tilesX;
    int tilesY;
    int* tileOffsets;   // size tilesX*tilesY
    int* tileCounts;    // size tilesX*tilesY
    int* tileIndices;   // concatenated indices, size = sum(tileCounts)
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstRendererParams;

// read-only lookup tables used to quickly compute noise (needed by
// advanceAnimation for the snowflake scene)
__constant__ int    cuConstNoiseYPermutationTable[256];
__constant__ int    cuConstNoiseXPermutationTable[256];
__constant__ float  cuConstNoise1DValueTable[256];

// color ramp table needed for the color ramp lookup shader
#define COLOR_MAP_SIZE 5
__constant__ float  cuConstColorRamp[COLOR_MAP_SIZE][3];


// including parts of the CUDA code from external files to keep this
// file simpler and to seperate code that should not be modified
#include "noiseCuda.cu_inl"
#include "lookupColor.cu_inl"
#include "circleBoxTest.cu_inl"

#define TILE_SIZE 32

// kernelClearImageSnowflake -- (CUDA device code)
//
// Clear the image, setting the image to the white-gray gradation that
// is used in the snowflake image
__global__ void kernelClearImageSnowflake() {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float shade = .4f + .45f * static_cast<float>(height-imageY) / height;
    float4 value = make_float4(shade, shade, shade, 1.f);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelClearImage --  (CUDA device code)
//
// Clear the image, setting all pixels to the specified color rgba
__global__ void kernelClearImage(float r, float g, float b, float a) {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r, g, b, a);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// Utility function to compute clamped screen bbox for a circle in pixel coords
__host__ __device__ __forceinline__ void circleScreenBBox(int imageWidth, int imageHeight, float3 p, float rad,
                                                 int* outMinX, int* outMaxX, int* outMinY, int* outMaxY) {
    float minXf = p.x - rad;
    float maxXf = p.x + rad;
    float minYf = p.y - rad;
    float maxYf = p.y + rad;
    int minX = static_cast<int>(minXf * imageWidth);
    int maxX = static_cast<int>(maxXf * imageWidth) + 1;
    int minY = static_cast<int>(minYf * imageHeight);
    int maxY = static_cast<int>(maxYf * imageHeight) + 1;
    // clamp
    minX = max(0, min(minX, imageWidth));
    maxX = max(0, min(maxX, imageWidth));
    minY = max(0, min(minY, imageHeight));
    maxY = max(0, min(maxY, imageHeight));
    *outMinX = minX; *outMaxX = maxX; *outMinY = minY; *outMaxY = maxY;
}

// kernelAdvanceFireWorks
// 
// Update the position of the fireworks (if circle is firework)
__global__ void kernelAdvanceFireWorks() {
    const float dt = 1.f / 60.f;
    const float pi = 3.14159;
    const float maxDist = 0.25f;

    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;
    float* radius = cuConstRendererParams.radius;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles)
        return;

    if (0 <= index && index < NUM_FIREWORKS) { // firework center; no update 
        return;
    }

    // determine the fire-work center/spark indices
    int fIdx = (index - NUM_FIREWORKS) / NUM_SPARKS;
    int sfIdx = (index - NUM_FIREWORKS) % NUM_SPARKS;

    int index3i = 3 * fIdx;
    int sIdx = NUM_FIREWORKS + fIdx * NUM_SPARKS + sfIdx;
    int index3j = 3 * sIdx;

    float cx = position[index3i];
    float cy = position[index3i+1];

    // update position
    position[index3j] += velocity[index3j] * dt;
    position[index3j+1] += velocity[index3j+1] * dt;

    // fire-work sparks
    float sx = position[index3j];
    float sy = position[index3j+1];

    // compute vector from firework-spark
    float cxsx = sx - cx;
    float cysy = sy - cy;

    // compute distance from fire-work 
    float dist = sqrt(cxsx * cxsx + cysy * cysy);
    if (dist > maxDist) { // restore to starting position 
        // random starting position on fire-work's rim
        float angle = (sfIdx * 2 * pi)/NUM_SPARKS;
        float sinA = sin(angle);
        float cosA = cos(angle);
        float x = cosA * radius[fIdx];
        float y = sinA * radius[fIdx];

        position[index3j] = position[index3i] + x;
        position[index3j+1] = position[index3i+1] + y;
        position[index3j+2] = 0.0f;

        // travel scaled unit length 
        velocity[index3j] = cosA/5.0;
        velocity[index3j+1] = sinA/5.0;
        velocity[index3j+2] = 0.0f;
    }
}

// kernelAdvanceHypnosis   
//
// Update the radius/color of the circles
__global__ void kernelAdvanceHypnosis() { 
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles) 
        return; 

    float* radius = cuConstRendererParams.radius; 

    float cutOff = 0.5f;
    // place circle back in center after reaching threshold radisus 
    if (radius[index] > cutOff) { 
        radius[index] = 0.02f; 
    } else { 
        radius[index] += 0.01f; 
    }   
}   


// kernelAdvanceBouncingBalls
// 
// Update the positino of the balls
__global__ void kernelAdvanceBouncingBalls() { 
    const float dt = 1.f / 60.f;
    const float kGravity = -2.8f; // sorry Newton
    const float kDragCoeff = -0.8f;
    const float epsilon = 0.001f;

    int index = blockIdx.x * blockDim.x + threadIdx.x; 
   
    if (index >= cuConstRendererParams.numCircles) 
        return; 

    float* velocity = cuConstRendererParams.velocity; 
    float* position = cuConstRendererParams.position; 

    int index3 = 3 * index;
    // reverse velocity if center position < 0
    float oldVelocity = velocity[index3+1];
    float oldPosition = position[index3+1];

    if (oldVelocity == 0.f && oldPosition == 0.f) { // stop-condition 
        return;
    }

    if (position[index3+1] < 0 && oldVelocity < 0.f) { // bounce ball 
        velocity[index3+1] *= kDragCoeff;
    }

    // update velocity: v = u + at (only along y-axis)
    velocity[index3+1] += kGravity * dt;

    // update positions (only along y-axis)
    position[index3+1] += velocity[index3+1] * dt;

    if (fabsf(velocity[index3+1] - oldVelocity) < epsilon
        && oldPosition < 0.0f
        && fabsf(position[index3+1]-oldPosition) < epsilon) { // stop ball 
        velocity[index3+1] = 0.f;
        position[index3+1] = 0.f;
    }
}

// kernelAdvanceSnowflake -- (CUDA device code)
//
// move the snowflake animation forward one time step.  Updates circle
// positions and velocities.  Note how the position of the snowflake
// is reset if it moves off the left, right, or bottom of the screen.
__global__ void kernelAdvanceSnowflake() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    const float dt = 1.f / 60.f;
    const float kGravity = -1.8f; // sorry Newton
    const float kDragCoeff = 2.f;

    int index3 = 3 * index;

    float* positionPtr = &cuConstRendererParams.position[index3];
    float* velocityPtr = &cuConstRendererParams.velocity[index3];

    // loads from global memory
    float3 position = *((float3*)positionPtr);
    float3 velocity = *((float3*)velocityPtr);

    // hack to make farther circles move more slowly, giving the
    // illusion of parallax
    float forceScaling = fmin(fmax(1.f - position.z, .1f), 1.f); // clamp

    // add some noise to the motion to make the snow flutter
    float3 noiseInput;
    noiseInput.x = 10.f * position.x;
    noiseInput.y = 10.f * position.y;
    noiseInput.z = 255.f * position.z;
    float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
    noiseForce.x *= 7.5f;
    noiseForce.y *= 5.f;

    // drag
    float2 dragForce;
    dragForce.x = -1.f * kDragCoeff * velocity.x;
    dragForce.y = -1.f * kDragCoeff * velocity.y;

    // update positions
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;

    // update velocities
    velocity.x += forceScaling * (noiseForce.x + dragForce.y) * dt;
    velocity.y += forceScaling * (kGravity + noiseForce.y + dragForce.y) * dt;

    float radius = cuConstRendererParams.radius[index];

    // if the snowflake has moved off the left, right or bottom of
    // the screen, place it back at the top and give it a
    // pseudorandom x position and velocity.
    if ( (position.y + radius < 0.f) ||
         (position.x + radius) < -0.f ||
         (position.x - radius) > 1.f)
    {
        noiseInput.x = 255.f * position.x;
        noiseInput.y = 255.f * position.y;
        noiseInput.z = 255.f * position.z;
        noiseForce = cudaVec2CellNoise(noiseInput, index);

        position.x = .5f + .5f * noiseForce.x;
        position.y = 1.35f + radius;

        // restart from 0 vertical velocity.  Choose a
        // pseudo-random horizontal velocity.
        velocity.x = 2.f * noiseForce.y;
        velocity.y = 0.f;
    }

    // store updated positions and velocities to global memory
    *((float3*)positionPtr) = position;
    *((float3*)velocityPtr) = velocity;
}

// MARK: Pixel shading generics

struct SnowflakePixelShader {
    __device__ void operator()(float3* rgb, float* alpha, int circleIndex, float3 p, float pixelDist, float rad) {
        const float kCircleMaxAlpha = .5f;
        const float falloffScale = 4.f;

        float normPixelDist = sqrt(pixelDist) / rad;
        *rgb = lookupColor(normPixelDist);

        float maxAlpha = .6f + .4f * (1.f-p.z);
        maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
        *alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);
    }
};

struct SimplePixelShader {
    __device__ void operator()(float3* rgb, float* alpha, int circleIndex, float3 p, float pixelDist, float rad) {
        // simple: each circle has an assigned color
        int index3 = 3 * circleIndex;
        *rgb = *(float3*)&(cuConstRendererParams.color[index3]);
        *alpha = .5f;
    }
};


// kernelRenderCircles -- (CUDA device code)
//
// tiled renderer using pre-binned per-tile circle lists
__global__ void kernelRenderCircles() {
    const int tileX = blockIdx.x;
    const int tileY = blockIdx.y;
    const int width = cuConstRendererParams.imageWidth;
    const int height = cuConstRendererParams.imageHeight;
    const int pixelX = tileX * TILE_SIZE + threadIdx.x;
    const int pixelY = tileY * TILE_SIZE + threadIdx.y;

    if (pixelX >= width || pixelY >= height) return;

    const float invWidth = 1.f / width;
    const float invHeight = 1.f / height;
    const float2 pixelCenter = make_float2((pixelX + 0.5f) * invWidth, (pixelY + 0.5f) * invHeight);

    const int tileId = tileY * cuConstRendererParams.tilesX + tileX;
    const int count = cuConstRendererParams.tileCounts[tileId];
    
    // Early exit if no circles in tile
    if (count == 0) return;
    
    const int offset = cuConstRendererParams.tileOffsets[tileId];

    const int imgOffset = 4 * (pixelY * width + pixelX);
    float4* imagePtr = (float4*)&cuConstRendererParams.imageData[imgOffset];
    float4 pixelColor = *imagePtr;

    // Shared memory for cooperative loading of circle attributes
    const int CHUNK = 1024;  // Match total thread count to minimize syncs
    __shared__ float shPosX[CHUNK];
    __shared__ float shPosY[CHUNK];
    __shared__ float shPosZ[CHUNK];
    __shared__ float shRad[CHUNK];
    __shared__ float shRadSq[CHUNK];
    __shared__ float shColorR[CHUNK];
    __shared__ float shColorG[CHUNK];
    __shared__ float shColorB[CHUNK];
    
    const int tid = threadIdx.y * TILE_SIZE + threadIdx.x;
    const int* indices = cuConstRendererParams.tileIndices;

    // Process circles in batches to maximize shared memory reuse
    for (int base = 0; base < count; base += CHUNK) {
        const int chunk = min(CHUNK, count - base);
        
        // Cooperative load: each thread loads one circle's attributes
        if (tid < chunk) {
            const int circleIndex = indices[offset + base + tid];
            const int index3 = 3 * circleIndex;
            
            // Use read-only cache loads for better memory performance
            const float px = __ldg(&cuConstRendererParams.position[index3]);
            const float py = __ldg(&cuConstRendererParams.position[index3 + 1]);
            const float pz = __ldg(&cuConstRendererParams.position[index3 + 2]);
            const float rad = __ldg(&cuConstRendererParams.radius[circleIndex]);
            const float cr = __ldg(&cuConstRendererParams.color[index3]);
            const float cg = __ldg(&cuConstRendererParams.color[index3 + 1]);
            const float cb = __ldg(&cuConstRendererParams.color[index3 + 2]);
            
            shPosX[tid] = px;
            shPosY[tid] = py;
            shPosZ[tid] = pz;
            shRad[tid] = rad;
            shRadSq[tid] = rad * rad;  // Precompute for inner loop
            shColorR[tid] = cr;
            shColorG[tid] = cg;
            shColorB[tid] = cb;
        }
        __syncthreads();

        // Shade this pixel using the loaded batch
        #pragma unroll 4
        for (int j = 0; j < chunk; ++j) {
            const float dx = shPosX[j] - pixelCenter.x;
            const float dy = shPosY[j] - pixelCenter.y;
            const float pixelDist = dx * dx + dy * dy;
            
            if (pixelDist <= shRadSq[j]) {
                const int circleIndex = indices[offset + base + j];
                float3 rgb;
                float alpha;
                
                if (cuConstRendererParams.sceneName == SNOWFLAKES || 
                    cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {
                    float3 p = make_float3(shPosX[j], shPosY[j], shPosZ[j]);
                    SnowflakePixelShader pixelShader;
                    pixelShader(&rgb, &alpha, circleIndex, p, pixelDist, shRad[j]);
                } else {
                    // Simple shader: use cached color
                    rgb = make_float3(shColorR[j], shColorG[j], shColorB[j]);
                    alpha = 0.5f;
                }
                
                // Alpha blending (optimized with FMA)
                const float oneMinusAlpha = 1.f - alpha;
                pixelColor.x = fmaf(alpha, rgb.x, oneMinusAlpha * pixelColor.x);
                pixelColor.y = fmaf(alpha, rgb.y, oneMinusAlpha * pixelColor.y);
                pixelColor.z = fmaf(alpha, rgb.z, oneMinusAlpha * pixelColor.z);
                pixelColor.w += alpha;
            }
        }
        __syncthreads();
    }

    *imagePtr = pixelColor;
}

////////////////////////////////////////////////////////////////////////////////////////


CudaRenderer::CudaRenderer() {
    image = NULL;

    numCircles = 0;
    position = NULL;
    velocity = NULL;
    color = NULL;
    radius = NULL;

    cudaDevicePosition = NULL;
    cudaDeviceVelocity = NULL;
    cudaDeviceColor = NULL;
    cudaDeviceRadius = NULL;
    cudaDeviceImageData = NULL;
    // bins will be built on first render
    binsDirty = true;
}

CudaRenderer::~CudaRenderer() {

    if (image) {
        delete image;
    }

    if (position) {
        delete [] position;
        delete [] velocity;
        delete [] color;
        delete [] radius;
    }

    if (cudaDevicePosition) {
        cudaFree(cudaDevicePosition);
        cudaFree(cudaDeviceVelocity);
        cudaFree(cudaDeviceColor);
        cudaFree(cudaDeviceRadius);
        cudaFree(cudaDeviceImageData);
    }
}

const Image*
CudaRenderer::getImage() {

    // need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    printf("Copying image data from device\n");

    cudaMemcpy(image->data,
               cudaDeviceImageData,
               sizeof(float) * 4 * image->width * image->height,
               cudaMemcpyDeviceToHost);

    return image;
}

void
CudaRenderer::loadScene(SceneName scene, int seed) {
    sceneName = scene;
    loadCircleScene(sceneName, numCircles, position, velocity, color, radius, seed);
    binsDirty = true;
    lastSceneName = sceneName;

}

void
CudaRenderer::setup() {

    int deviceCount = 0;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
    
    // By this time the scene should be loaded.  Now copy all the key
    // data structures into device memory so they are accessible to
    // CUDA kernels
    //
    // See the CUDA Programmer's Guide for descriptions of
    // cudaMalloc and cudaMemcpy

    cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceRadius, sizeof(float) * numCircles);
    cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);

    cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numCircles, cudaMemcpyHostToDevice);

    // Initialize parameters in constant memory.  We didn't talk about
    // constant memory in class, but the use of read-only constant
    // memory here is an optimization over just sticking these values
    // in device global memory.  NVIDIA GPUs have a few special tricks
    // for optimizing access to constant memory.  Using global memory
    // here would have worked just as well.  See the Programmer's
    // Guide for more information about constant memory.

    GlobalConstants params; // zero-init all fields to avoid garbage pointers
    memset(&params, 0, sizeof(GlobalConstants));
    params.sceneName = sceneName;
    params.numCircles = numCircles;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.position = cudaDevicePosition;
    params.velocity = cudaDeviceVelocity;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.imageData = cudaDeviceImageData;

    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

    // also need to copy over the noise lookup tables, so we can
    // implement noise on the GPU
    int* permX;
    int* permY;
    float* value1D;
    getNoiseTables(&permX, &permY, &value1D);
    cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

    // last, copy over the color table that's used by the shading
    // function for circles in the snowflake demo

    float lookupTable[COLOR_MAP_SIZE][3] = {
        {1.f, 1.f, 1.f},
        {1.f, 1.f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, 0.8f, 1.f},
    };

    cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);

}

// allocOutputImage --
//
// Allocate buffer the renderer will render into.  Check status of
// image first to avoid memory leak.
void
CudaRenderer::allocOutputImage(int width, int height) {

    if (image)
        delete image;
    image = new Image(width, height);
    binsDirty = true; // image dimensions changed -> tiles change
}

// clearImage --
//
// Clear's the renderer's target image.  The state of the image after
// the clear depends on the scene being rendered.
void
CudaRenderer::clearImage() {

    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        kernelClearImageSnowflake<<<gridDim, blockDim>>>();
    } else {
        kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    }
    cudaDeviceSynchronize();
}

// advanceAnimation --
//
// Advance the simulation one time step.  Updates all circle positions
// and velocities
void
CudaRenderer::advanceAnimation() {
     // 256 threads per block is a healthy number
    dim3 blockDim(256, 1);
    dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

    // only the snowflake scene has animation
    if (sceneName == SNOWFLAKES) {
        kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
    } else if (sceneName == BOUNCING_BALLS) {
        kernelAdvanceBouncingBalls<<<gridDim, blockDim>>>();
    } else if (sceneName == HYPNOSIS) { 
        kernelAdvanceHypnosis<<<gridDim, blockDim>>>(); 
    } else if (sceneName == FIREWORKS) {
        kernelAdvanceFireWorks<<<gridDim, blockDim>>>(); 
    }
    cudaDeviceSynchronize();
    // positions/radii potentially changed
    binsDirty = true;
}

void
CudaRenderer::render() {
    int tilesX = (image->width + TILE_SIZE - 1) / TILE_SIZE;
    int tilesY = (image->height + TILE_SIZE - 1) / TILE_SIZE;
    int numTiles = tilesX * tilesY;

    // allocate/resize temporary device buffers lazily via static locals
    static int allocatedTilesAlloc = 0; // number of tiles allocated for counts/offsets
    static int* d_tileCounts = nullptr;
    static int* d_tileOffsets = nullptr;
    static int* d_tileIndices = nullptr;
    static size_t allocatedIndices = 0;

    if (numTiles != allocatedTilesAlloc) {
        if (d_tileCounts) { 
            cudaFree(d_tileCounts);
            cudaFree(d_tileOffsets);
            
            if (d_tileIndices)
                cudaFree(d_tileIndices);
            
            d_tileIndices = nullptr;
            allocatedIndices = 0;
        }
        
        cudaMalloc(&d_tileCounts, sizeof(int) * numTiles);
        cudaMalloc(&d_tileOffsets, sizeof(int) * numTiles);
        allocatedTilesAlloc = numTiles;
    }

    // Only build bins when necessary: after animation or changes
    static bool binsBuilt = false;
    if (binsDirty || !binsBuilt || lastTilesX != tilesX || lastTilesY != tilesY || lastNumCircles != numCircles || lastSceneName != sceneName) {
        size_t total = 0;
        // Host-side binning per tile
        cudaMemcpy(position, cudaDevicePosition, sizeof(float) * 3 * numCircles, cudaMemcpyDeviceToHost);
        cudaMemcpy(radius,   cudaDeviceRadius,   sizeof(float) * numCircles,     cudaMemcpyDeviceToHost);

        std::vector<int> h_counts(numTiles, 0);
        std::vector<int> h_offsets(numTiles, 0);
        for (int circleIndex = 0; circleIndex < numCircles; ++circleIndex) {
            int index3 = 3 * circleIndex;
            float3 p = *(float3*)&position[index3];
            float rad = radius[circleIndex];
            int minX, maxX, minY, maxY;
            circleScreenBBox(image->width, image->height, p, rad, &minX, &maxX, &minY, &maxY);
            
            int tileMinX = minX / TILE_SIZE;
            int tileMaxX = (maxX + TILE_SIZE - 1) / TILE_SIZE;
            int tileMinY = minY / TILE_SIZE;
            int tileMaxY = (maxY + TILE_SIZE - 1) / TILE_SIZE;
            tileMinX = std::max(0, std::min(tileMinX, tilesX));
            tileMaxX = std::max(0, std::min(tileMaxX, tilesX));
            tileMinY = std::max(0, std::min(tileMinY, tilesY));
            tileMaxY = std::max(0, std::min(tileMaxY, tilesY));
            
            for (int ty = tileMinY; ty < tileMaxY; ++ty) {
                int rowBase = ty * tilesX;
                for (int tx = tileMinX; tx < tileMaxX; ++tx) {
                    h_counts[rowBase + tx]++;
                }
            }
        }

        total = 0;
        for (int i = 0; i < numTiles; ++i) {
            int c = h_counts[i];
            h_offsets[i] = total;
            total += c;
        }
        
        if (total > allocatedIndices) {
            if (d_tileIndices)
                cudaFree(d_tileIndices);
            
            cudaMalloc(&d_tileIndices, sizeof(int) * total);
            allocatedIndices = total;
        }
        
        std::vector<int> h_indices(total);
        std::vector<int> h_write = h_offsets;
        for (int circleIndex = 0; circleIndex < numCircles; ++circleIndex) {
            int index3 = 3 * circleIndex;
            float3 p = *(float3*)&position[index3];
            float rad = radius[circleIndex];
            int minX, maxX, minY, maxY;
            circleScreenBBox(image->width, image->height, p, rad, &minX, &maxX, &minY, &maxY);
            
            int tileMinX = minX / TILE_SIZE;
            int tileMaxX = (maxX + TILE_SIZE - 1) / TILE_SIZE;
            int tileMinY = minY / TILE_SIZE;
            int tileMaxY = (maxY + TILE_SIZE - 1) / TILE_SIZE;
            tileMinX = std::max(0, std::min(tileMinX, tilesX));
            tileMaxX = std::max(0, std::min(tileMaxX, tilesX));
            tileMinY = std::max(0, std::min(tileMinY, tilesY));
            tileMaxY = std::max(0, std::min(tileMaxY, tilesY));
            
            for (int ty = tileMinY; ty < tileMaxY; ++ty) {
                int rowBase = ty * tilesX;
                for (int tx = tileMinX; tx < tileMaxX; ++tx) {
                    int tileId = rowBase + tx;
                    int pos = h_write[tileId]++;
                    h_indices[pos] = circleIndex;
                }
            }
        }
        cudaMemcpy(d_tileCounts, h_counts.data(), sizeof(int) * numTiles, cudaMemcpyHostToDevice);
        cudaMemcpy(d_tileOffsets, h_offsets.data(), sizeof(int) * numTiles, cudaMemcpyHostToDevice);
        cudaMemcpy(d_tileIndices, h_indices.data(), sizeof(int) * total,    cudaMemcpyHostToDevice);

        // mark cache state
        binsDirty = false; binsBuilt = true;
        lastTilesX = tilesX; lastTilesY = tilesY; lastNumCircles = numCircles; lastSceneName = sceneName;
    }

    // Update constant params for tile rendering
    GlobalConstants params;
    cudaMemcpyFromSymbol(&params, cuConstRendererParams, sizeof(GlobalConstants));
    params.tilesX = tilesX;
    params.tilesY = tilesY;
    params.tileCounts = d_tileCounts;
    params.tileOffsets = d_tileOffsets;
    params.tileIndices = d_tileIndices;
    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

    // Render per tile
    dim3 blockDim(TILE_SIZE, TILE_SIZE, 1);
    dim3 gridDim(tilesX, tilesY);
    kernelRenderCircles<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();
}
