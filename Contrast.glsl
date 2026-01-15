#version 460
// Contrast GLSL Library - Apache 2.0 - https://www.apache.org/licenses/LICENSE-2.0

#ifdef GL
precision mediump float;
#endif

layout(location = 0) out vec4 fragColor;

uniform float u_time;
uniform vec2 u_resolution;
uniform vec2 u_mouse;


/***********************
    SCALAR CONSTANTS
***********************/

const float PI = 3.14159265359;
const float TAU = 6.28318530717;

const float FLOAT_ERROR = 1e-4;
const float QSDF_MIN_GRAD = 1e-5;
const float FLT_MAX = 3.402823466e+38;


const mat2 MATH_IDENTITY_MATRIX_2D = mat2(1.0, 0.0,
                                          0.0, 1.0);

const mat3 MATH_IDENTITY_MATRIX_3D = mat3(1.0, 0.0, 0.0,
                                          0.0, 1.0, 0.0,
                                          0.0, 0.0, 1.0);

const mat4 MATH_IDENTITY_MATRIX_4D = mat4(1.0, 0.0, 0.0, 0.0,
                                          0.0, 1.0, 0.0, 0.0,
                                          0.0, 0.0, 1.0, 0.0,
                                          0.0, 0.0, 0.0, 1.0);


/***********************
    TEMPORAL CONTROL
***********************/

const struct TIME_StageConfig{
    float stageLength;
    uint stages;
    uint frames;
    float loopLength;
    float frameLength;
};

struct TIME_StageInfo{
    float loopTime;
    uint currentStage;
    float stageProgress;
    float stageTime;
    float stageFrame;
};

TIME_StageConfig TIME_SetStageConfig(float stageLength, uint stages, uint frames){
    return TIME_StageConfig(stageLength, stages, frames, stageLength*float(stages), stageLength/float(frames));
}

TIME_StageInfo TIME_GetStageInfo(float time, TIME_StageConfig config){
    float loopTime = mod(time, config.loopLength);
    uint currentStage = uint(loopTime/config.stageLength);
    float stageProgress = fract(loopTime/config.stageLength);
    float stageTime = mod(loopTime, config.stageLength);
    float stageFrame = stageTime/config.frames;
    return TIME_StageInfo(loopTime, currentStage, stageProgress, stageTime, stageFrame);
}

bool TIME_IsStage(uint currentStage, uint targetStage){
    return currentStage == targetStage;
}

bool TIME_IsInStageRange(uint currentStage, uint startStage, uint endStage){
    return (currentStage >= startStage) && (currentStage < endStage);
}

bool TIME_IsNotInStageRange(uint currentStage, uint lowerStage, uint higherStage){
    return (currentStage < lowerStage) || (currentStage > higherStage);
}

bool TIME_IsAfterStage(uint currentStage, uint startStage){
    return (currentStage > startStage);
}

bool TIME_IsBeforeStage(uint currentStage, uint startStage){
    return (currentStage < startStage);
}

bool TIME_IsInSubstageRange(float stageFrame, float lowerFrame, float higherFrame){
    return (stageFrame >= lowerFrame) && (stageFrame < higherFrame);
}

bool TIME_IsNotInSubstageRange(float stageFrame, float lowerFrame, float higherFrame){
    return (stageFrame < lowerFrame) || (stageFrame > higherFrame);
}

bool TIME_IsBeforeStageFrame(float stageFrame, float boundFrame){
    return (stageFrame < boundFrame);
}

bool TIME_IsAfterStageFrame(float stageFrame, float boundFrame){
    return (stageFrame > boundFrame);
}


/***********************
    POLAR COORDINATES
***********************/

struct MATH_PolarCoords {
    float r;
    float a;
};

MATH_PolarCoords MATH_GetUV2Polar(vec2 uv){
    MATH_PolarCoords p;
    p.r = length(uv);
    p.a = atan(uv.y, uv.x);
    return p;
}


/***********************
 LINEAR TRANSFORMATIONS
***********************/

struct MATH_Transform2D{
    mat2 rotation;
    vec2 translation;
};

MATH_Transform2D MATH_SetTransform(mat2 rotation, vec2 translation){
    return MATH_Transform2D(rotation, translation);
}

MATH_Transform2D MATH_SetTransform(vec2 translation){
    return MATH_Transform2D(MATH_IDENTITY_MATRIX_2D, translation);
}

MATH_Transform2D MATH_SetTransform(mat2 rotation){
    return MATH_Transform2D(rotation, vec2(0.0));
}


struct MATH_Transform3D{
    mat3 rotation;
    vec3 translation;
};

MATH_Transform3D MATH_SetTransform(mat3 rotation, vec3 translation){
    return MATH_Transform3D(rotation, translation);
}

MATH_Transform3D MATH_SetTransform(mat3 rotation){
    return MATH_Transform3D(rotation, vec3(0.0));
}

MATH_Transform3D MATH_SetTransform(vec3 translation){
    return MATH_Transform3D(MATH_IDENTITY_MATRIX_3D, translation);
}

MATH_Transform2D MATH_GetTransformInverse(MATH_Transform2D T){
    MATH_Transform2D invT;
    invT.rotation = transpose(T.rotation);
    invT.translation = -T.translation;
    return invT;
}

MATH_Transform3D MATH_GetTransformInverse(MATH_Transform3D T){
    MATH_Transform3D invT;
    invT.rotation = transpose(T.rotation);
    invT.translation = -T.translation;
    return invT;
}

vec2 MATH_ApplyTransform(vec2 uv, MATH_Transform2D T){
    vec2 uv0 = uv + T.translation;
    uv0 = T.rotation*uv0;
    return uv0;
}

vec2 MATH_ApplyTransform2(vec2 uv, MATH_Transform2D T){
    vec2 uv0 = T.rotation*uv;
    uv0 += T.translation;
    return uv0;
}

float MATH_ScalarProj(vec2 a, vec2 b){
    return dot(a, b)/dot(b, b);
}

float MATH_Cross(vec2 a, vec2 b){
    return a.x*b.y - a.y*b.x;
}

mat2 MATH_Rotation2D(float angle){
    return mat2(cos(angle), sin(angle),
                -sin(angle), cos(angle));
}

float MATH_Remap(float x, float lowIn, float highIn, float lowOut, float highOut){
    return (x - lowIn) / (highIn - lowIn) * (highOut - lowOut) + lowOut;
}

float MATH_Quantize(float x, float step){
    return floor(x * step)/step;
}

vec2 MATH_Quantize(vec2 x, float step){
    return floor(x * step)/step;
}

vec3 MATH_Quantize(vec3 x, float step){
    return floor(x * step)/step;
}



float MATH_Remap01(float x, float lowIn, float highIn){
    return (x - lowIn) / (highIn - lowIn);
}

float MATH_NormalDistribution(float value){
    return exp(-value*value) / sqrt(2*PI);
}




/***********************
          SDFs 
***********************/

struct SDF_Band {
    float halfSize;
    MATH_Transform2D invTransform;
};

SDF_Band SDF_SetInfoBand(float halfSize, MATH_Transform2D invTransform){
    SDF_Band p;
    p.halfSize = halfSize;
    p.invTransform = invTransform;
    return p;
}

float SDF_EvalBand(vec2 uv, SDF_Band band){
    vec2 uv0 = MATH_ApplyTransform(uv, band.invTransform);
    return abs(uv0.y) - band.halfSize;
}


struct SDF_Box{
    vec2 halfSize;
    float cornerRadius;
    MATH_Transform2D invTransform;
};

SDF_Box SDF_SetInfoBox(vec2 halfSize, float cornerRadius, MATH_Transform2D invTransform){
    SDF_Box p;
    p.halfSize = halfSize;
    p.cornerRadius = cornerRadius;
    p.invTransform = invTransform;
    return p;
}

float SDF_EvalBox(vec2 uv, SDF_Box p){
    vec2 uv0 = MATH_ApplyTransform(uv, p.invTransform);
    vec2 hyperband = abs(uv) - p.halfSize + p.cornerRadius;
    return length(max(hyperband, 0.0)) + min(max(hyperband.x, hyperband.y), 0.0) - p.cornerRadius;
}


struct SDF_Circle{
    float radius;
    MATH_Transform2D invTransform;
};

SDF_Circle SDF_SetInfoCircle(float radius, MATH_Transform2D invTransform){
    SDF_Circle p;
    p.radius = radius;
    p.invTransform = invTransform;
    return p;
}

float SDF_EvalCircle(vec2 uv, SDF_Circle c){
    vec2 uv0 = MATH_ApplyTransform(uv, c.invTransform);
    return length(uv0) - c.radius;
}


struct SDF_RingCircle{
    float radius;
    float thickness;
    MATH_Transform2D invTransform;
};

float SDF_EvalRing(vec2 uv, SDF_RingCircle p){
    vec2 uv0 = MATH_ApplyTransform(uv, p.invTransform);
    return abs(length(uv0) - p.radius + p.thickness) - p.thickness;
}

SDF_RingCircle SDF_SetInfoRing(float radius, float thickness, MATH_Transform2D invTransform){
    SDF_RingCircle p;
    p.radius = radius;
    p.thickness = thickness;
    p.invTransform = invTransform;
    return p;
}


struct SDF_NAgon{
    float size;
    float sides;
    MATH_Transform2D invTransform;
};

SDF_NAgon SDF_SetInfoNAgon(float size, float sides, MATH_Transform2D invTransform){
    SDF_NAgon p;
    p.size = size;
    p.sides = sides;
    p.invTransform = invTransform;
    return p;
}

float SDF_EvalNAgon(vec2 uv, SDF_NAgon p){
    vec2 uv0 = MATH_ApplyTransform(uv, p.invTransform);
    float uvAngle = atan(uv0.y, uv0.x);
    float uvRadius = length(uv0);
    float halfSliceAngle = PI / p.sides;

    float sliceAngle = TAU / p.sides;;
    float radiusCorrection = cos(halfSliceAngle);

    float localAngle = mod(uvAngle + halfSliceAngle, sliceAngle) - halfSliceAngle;
    float distance = cos(localAngle)*uvRadius;
    return distance - p.size*radiusCorrection;
}


struct SDF_Triangle{
    mat3x2 vertex;
    MATH_Transform2D invTransform;
};

SDF_Triangle SDF_SetInfoTriangle(mat3x2 vertex, MATH_Transform2D invTransform){
    SDF_Triangle p;
    p.vertex = vertex;
    p.invTransform = invTransform;
    return p;
}

float SDF_EvalTriangle(vec2 uv, SDF_Triangle p){
    mat3x2 vertex = p.vertex;
    vec2 uv0 = MATH_ApplyTransform(uv, p.invTransform);
    int nVertex = 3;
    vec2 vect, pq;
    mat3x2 edges, gates;
    vec2 d = vec2(1e20);
    for (int i = 0; i < nVertex; i++){
        edges[i] = vertex[(i+1)%nVertex]-vertex[i];
        vect = uv0-vertex[i];
        pq = vect - edges[i]*clamp(MATH_ScalarProj(vect, edges[i]), 0.0, 1.0);
        gates[i] = vec2(dot(pq,pq), MATH_Cross(vect, edges[i]));
    }
    float s = sign(MATH_Cross(edges[0], edges[2]));

    for(int i = 0; i < nVertex; i++){
        gates[i].y *= s;
        d = min(d, gates[i]);
    }
    return -sqrt(d.x)*sign(d.y);
}


struct SDF_RoundedBox{
    vec2 halfSize;
    float cornerRadius;
    MATH_Transform2D invTransform;
};



/***********************
       QUASI-SDFs 
***********************/



struct QSDF_Line2D{
    float slope;
    MATH_Transform2D invTransform;
};

QSDF_Line2D QSDF_SetInfoLine2D(float slope, MATH_Transform2D invTransform){
    QSDF_Line2D p;
    p.slope = slope;
    p.invTransform = invTransform;
    return p;
};

float QSDF_EvalLine2D(vec2 uv, QSDF_Line2D p){
    vec2 uv0 = MATH_ApplyTransform(uv, p.invTransform);
    return (p.slope*uv.x-uv.y);
}

float QSDF_ToDistance(float field){
    float eps_s = 2.0/u_resolution.y;
    float grad = min(fwidth(field), QSDF_MIN_GRAD)/eps_s;
    return field/grad;
}

float idk(vec2 uv){
    return (uv.x*uv.x-uv.y);
}


/***********************
    MASK EVALUATION 
***********************/

struct MASK_Profile{
    float field;
    float blur;
    float offset;
};

MASK_Profile MASK_SetInfoProfile(float field, float blur, float offset){
    MASK_Profile p;
    p.field = field;
    p.blur = blur;
    p.offset = offset;
    return p;
};

float MASK_SoftEdge(MASK_Profile m){
    return smoothstep(-m.blur, m.blur, m.offset - m.field);
}

float MASK_HardEdge(MASK_Profile m){
    return step(0.0, m.offset - m.field);
}

/***********************
    COLOR OPERATIONS 
***********************/


vec3 COLOR_GammaCurve(vec3 color, vec3 gamma){
    return pow(color, gamma);
}

vec3 COLOR_PaletteCosine(vec3 brightness, vec3 contrast, vec3 amplitude, vec3 hue, float t){
    return brightness + contrast*(cos(TAU*(amplitude*t + hue)));
}

vec3 COLOR_PaletteTurbo(vec3 brightness, vec3 contrast, vec3 amplitude, vec3 hue, float t){
    return brightness + contrast*(cos(TAU*(amplitude*t + hue)));
}

vec3 COLOR_PaletteViridis(vec3 brightness, vec3 contrast, vec3 amplitude, vec3 hue, float t){
    return brightness + contrast*(cos(TAU*(amplitude*t + hue)));
}
