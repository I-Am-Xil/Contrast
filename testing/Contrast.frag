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
const float FLOTA_MAX = 3.402823466e+38;


const vec2 MATH_DEFAULT_TRANSLATION_2D = vec2(0.0);
const vec3 MATH_DEFAULT_TRANSLATION_3D = vec3(0.0);

const mat2 MATH_IDENTITY_MATRIX_2D = mat2(1.0, 0.0,
                                          0.0, 1.0);

const mat3 MATH_IDENTITY_MATRIX_3D = mat3(1.0, 0.0, 0.0,
                                          0.0, 1.0, 0.0,
                                          0.0, 0.0, 1.0);

const mat4 MATH_IDENTITY_MATRIX_4D = mat4(1.0, 0.0, 0.0, 0.0,
                                          0.0, 1.0, 0.0, 0.0,
                                          0.0, 0.0, 1.0, 0.0,
                                          0.0, 0.0, 0.0, 1.0);

const uint MATH_TRANSFORM_ROT_TRANS = 0;
const uint MATH_TRANSFORM_TRANS_ROT = 1;

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
    uint order;
};

MATH_Transform2D MATH_SetTransform(mat2 rotation, vec2 translation, uint order){
    return MATH_Transform2D(rotation, translation, order);
}

MATH_Transform2D MATH_SetTransform(vec2 translation, uint order){
    return MATH_Transform2D(MATH_IDENTITY_MATRIX_2D, translation, order);
}

MATH_Transform2D MATH_SetTransform(vec2 translation){
    return MATH_Transform2D(MATH_IDENTITY_MATRIX_2D, translation, MATH_TRANSFORM_ROT_TRANS);
}

MATH_Transform2D MATH_SetTransform(mat2 rotation, vec2 translation){
    return MATH_Transform2D(rotation, translation, MATH_TRANSFORM_ROT_TRANS);
}

MATH_Transform2D MATH_SetTransform(mat2 rotation){
    return MATH_Transform2D(rotation, MATH_DEFAULT_TRANSLATION_2D, MATH_TRANSFORM_ROT_TRANS);
}


struct MATH_Transform3D{
    mat3 rotation;
    vec3 translation;
    uint order;
};

MATH_Transform3D MATH_SetTransform(mat3 rotation, vec3 translation, uint order){
    return MATH_Transform3D(rotation, translation, order);
}

MATH_Transform3D MATH_SetTransform(vec3 translation, uint order){
    return MATH_Transform3D(MATH_IDENTITY_MATRIX_3D, translation, order);
}

MATH_Transform3D MATH_SetTransform(vec3 translation){
    return MATH_Transform3D(MATH_IDENTITY_MATRIX_3D, translation, MATH_TRANSFORM_ROT_TRANS);
}

MATH_Transform3D MATH_SetTransform(mat3 rotation, vec3 translation){
    return MATH_Transform3D(rotation, translation, MATH_TRANSFORM_ROT_TRANS);
}

MATH_Transform3D MATH_SetTransform(mat3 rotation){
    return MATH_Transform3D(rotation, MATH_DEFAULT_TRANSLATION_3D, MATH_TRANSFORM_ROT_TRANS);
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
    vec2 uv0 = uv;
    if (T.order == MATH_TRANSFORM_ROT_TRANS){
        uv0 += T.translation;
        uv0 = T.rotation*uv0;
        return uv0;
    }
    if (T.order == MATH_TRANSFORM_TRANS_ROT){
        uv0 = T.rotation*uv0;
        uv0 += T.translation;
        return uv0;
    }
}

vec3 MATH_ApplyTransform(vec3 uv, MATH_Transform3D T){
    vec3 uv0;
    if (T.order == MATH_TRANSFORM_ROT_TRANS){
        uv0 = uv + T.translation;
        uv0 = T.rotation*uv0;
        return uv0;
    }
    if (T.order == MATH_TRANSFORM_TRANS_ROT){
        uv0 = T.rotation*uv0;
        uv0 = uv + T.translation;
        return uv0;
    }
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

vec3 MATH_Pow(vec3 x, float r){
    return sign(x)*pow(abs(x), vec3(r));
}

vec3 MATH_Cbrt(vec3 x){
    return sign(x)*pow(abs(x), vec3(1.0/3.0));
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


float SDF_Intersection(float sdf1, float sdf2){
    return max(sdf1, sdf2);
}

float SDF_Union(float sdf1, float sdf2){
    return min(sdf1, sdf2);
}

float SDF_Subtraction(float sdf1, float sdf2){
    return max(-sdf1, sdf2);
}

float SDF_Xor(float sdf1, float sdf2){
    return max(min(sdf1, sdf2), -max(sdf1, sdf2));
}




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

//INFO: SRGB -> LRGB -> OKLAB -> LRGB -> SRGB

struct COLOR_SRGB{
    vec3 rgb;
};

struct COLOR_LSRGB{
    vec3 rgb;
};

COLOR_LSRGB COLOR_SRGBtoLRGB(COLOR_SRGB c){
    vec3 high = pow((c.rgb + 0.055) / 1.055, vec3(2.4));
    vec3 low = c.rgb / 12.92;
    vec3 mask = step(vec3(0.04045), c.rgb);
    return COLOR_LSRGB(mix(low, high, mask));
}

COLOR_SRGB COLOR_LSRGBtoSRGB(COLOR_LSRGB c){
    vec3 high = 1.055 * pow(c.rgb, vec3(1.0/2.4)) - 0.055;
    vec3 low = c.rgb * 12.92;
    vec3 mask = step(vec3(0.0031308), c.rgb);
    return COLOR_SRGB(mix(low, high, mask));
}

struct COLOR_XYZ{
    vec3 xyz;
};

struct COLOR_HSV{
    vec3 hsv;
};

struct COLOR_HSL{
    vec3 hsl;
};

struct COLOR_OKLAB{
    vec3 lab;
};

COLOR_OKLAB COLOR_LSRGBtoOKLAB(COLOR_LSRGB c){
    //NOTE: GLSL is column major. the m1 and m2 matrices are transposed to align with column representation.
    mat3 m1 = mat3(0.4122214708, 0.2119034982, 0.0883024619,
                   0.5363325363, 0.6806995451, 0.2817188376,
                   0.0514459929, 0.1073969566, 0.6299787005);
    vec3 lms = m1*c.rgb;
    vec3 lms_ = MATH_Cbrt(lms);
    mat3 m2 = mat3(0.2104542553,  1.9779984951,  0.0259040371,
                   0.7936177850, -2.4285922050,  0.7827717662,
                   -0.0040720468, 0.4505937099, -0.8086757660);
    return COLOR_OKLAB(m2*lms_);
}

COLOR_LSRGB COLOR_OKLABto_LSRGB(COLOR_OKLAB c){
    //NOTE: GLSL is column major. the m1 and m2 matrices are transposed to align with column representation.
    mat3 m1 = mat3(1.0,           1.0,           1.0,
                   0.3963377774, -0.1055613458, -0.0894841775,
                   0.2158037573, -0.0638541728, -1.2914855480);
    vec3 lms_ = m1*c.lab;
    vec3 lms = pow(lms_, vec3(3.0));
    mat3 m2 = mat3( 4.0767416621, -1.2684380046, -0.0041960863,
                   -3.3077115913,  2.6097574011, -0.7034186147,
                   0.2309699292, -0.3413193965,  1.7076147010);
    return COLOR_LSRGB(m2*lms);
}

struct COLOR_OKHSV{
    vec3 hsv;
};

struct COLOR_OKHSL{
    vec3 hsl;
};

COLOR_SRGB COLOR_Mix(COLOR_SRGB c, COLOR_SRGB c0, float a){
    return COLOR_SRGB(mix(c.rgb, c0.rgb, a));
}

COLOR_LSRGB COLOR_Mix(COLOR_LSRGB c, COLOR_LSRGB c0, float a){
    return COLOR_LSRGB(mix(c.rgb, c0.rgb, a));
}

COLOR_OKLAB COLOR_Mix(COLOR_OKLAB c, COLOR_OKLAB c0, float a){
    return COLOR_OKLAB(mix(c.lab, c0.lab, a));
}






vec3 COLOR_GammaCurve(vec3 color, vec3 gamma){
    return pow(color, gamma);
}

vec3 COLOR_PaletteCosine(vec3 brightness, vec3 contrast, vec3 amplitude, vec3 hue, float t){
    return brightness + contrast*(cos(TAU*(amplitude*t + hue)));
}

vec3 COLOR_MapTurbo(float t){
    const vec3 c0 = vec3(0.1140890109226559, 0.06288340699912215, 0.2248337216805064); const vec3 c1 = vec3(6.716419496985708, 3.182286745507602, 7.571581586103393); const vec3 c2 = vec3(-66.09402360453038, -4.9279827041226, -10.09439367561635); const vec3 c3 = vec3(228.7660791526501, 25.04986699771073, -91.54105330182436); const vec3 c4 = vec3(-334.8351565777451, -69.31749712757485, 288.5858850615712); const vec3 c5 = vec3(218.7637218434795, 67.52150567819112, -305.2045772184957); const vec3 c6 = vec3(-52.88903478218835, -21.54527364654712, 110.5174647748972);
    return c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6))))); // turbo
}

vec3 COLOR_MapViridis(float t){
    const vec3 c0 = vec3(0.2777273272234177, 0.005407344544966578, 0.3340998053353061); const vec3 c1 = vec3(0.1050930431085774, 1.404613529898575, 1.384590162594685); const vec3 c2 = vec3(-0.3308618287255563, 0.214847559468213, 0.09509516302823659); const vec3 c3 = vec3(-4.634230498983486, -5.799100973351585, -19.33244095627987); const vec3 c4 = vec3(6.228269936347081, 14.17993336680509, 56.69055260068105); const vec3 c5 = vec3(4.776384997670288, -13.74514537774601, -65.35303263337234); const vec3 c6 = vec3(-5.435455855934631, 4.645852612178535, 26.3124352495832);
    return c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6))))); // viridis
}

void main(){
    vec2 uv = (2.0 * gl_FragCoord.xy - u_resolution) / u_resolution.y;

    float shape = 0.0;

    MATH_Transform2D transform = MATH_SetTransform(MATH_Rotation2D(u_time), vec2(0.1));
    //transform = MATH_SetTransform(MATH_Rotation2D(u_time));
    transform = MATH_GetTransformInverse(transform);

    SDF_Band band = SDF_SetInfoBand(0.5, transform);
    shape = SDF_EvalBand(uv, band);

    SDF_Circle circle = SDF_SetInfoCircle(0.5, transform);
    shape = SDF_EvalCircle(uv, circle);

    SDF_RingCircle ring = SDF_SetInfoRing(0.5, 0.05, transform);
    shape = SDF_EvalRing(uv, ring);

    SDF_Box box = SDF_SetInfoBox(vec2(0.5), 0.0, transform);
    shape = SDF_EvalBox(uv, box);

    SDF_NAgon nAgon = SDF_SetInfoNAgon(0.5, 8.0, transform);
    shape = SDF_EvalNAgon(uv, nAgon);

    QSDF_Line2D line2d = QSDF_SetInfoLine2D(1.0, transform);
    shape = QSDF_EvalLine2D(uv, line2d);

    mat3x2 triangleVertex = mat3x2(0.0, 0.0,
                                   0.5, 0.0,
                                   0.0, 0.5);
    SDF_Triangle triangle = SDF_SetInfoTriangle(triangleVertex, transform);
    shape = SDF_EvalTriangle(uv, triangle);

    //shape = MATH_NormalDistribution(uv.x)-uv.y;

    //shape = SDF_Intersection(SDF_EvalCircle(uv, circle), SDF_EvalCircle(uv-0.1, circle));

    MASK_Profile mask = MASK_SetInfoProfile(shape, 0.01, 0.05);

    vec3 color = vec3(0.0);
    color = vec3(MASK_SoftEdge(mask));

    //vec3(0.556, 0.280, 0.738), vec3(0.941, 0.504, 0.558), vec3(0.935, 0.781, 0.071), vec3(5.902, 5.607, 2.693)

    //color = COLOR_PaletteCosine(vec3(0.556, 0.280, 0.738), vec3(0.941, 0.504, 0.558), vec3(0.935, 0.781, 0.071), vec3(5.902, 5.607, 2.693), uv.x+u_time);

    //color = COLOR_MapTurbo(0.5+uv.x);
    //color = COLOR_MapViridis(0.4+uv.x);

    //color = COLOR_Palette(vec3(0.556, 0.280, 0.738), vec3(0.941, 0.504, 0.558), vec3(0.935, 0.781, 0.071), vec3(5.902, 5.607, 2.693), uv.x+u_time);



    float gamma = 3.25 + 0.25*sin(u_time);
    //color = vec3(uv, 0.0);
    //color = FX_GammaCurve(color, gamma);
    //color = vec3(MASK_HardEdge(mask));
    //color = vec3(MASK_SoftEdge(mask));
    //color = vec3(shape);

    fragColor = vec4(color, 1.0);
}
