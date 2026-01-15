#version 460
// Contrast GLSL Library - Apache 2.0 - https://www.apache.org/licenses/LICENSE-2.0

#ifdef GL
precision mediump float;
#endif

out vec4 fragColor;

uniform float u_time;
uniform vec2 u_resolution;
uniform vec2 u_mouse;


vec2 MATH_Perpendicular(vec2 v){
    return vec2(-v.y, v.x);
}

float MATH_ScalarProjection(vec2 uv, vec2 direction){
    return dot(normalize(direction), uv);
}


float SDFU_SoftEdge(float sdf, float blur){
    return smoothstep(0, -blur, sdf);
}

vec2 SDFU_ScaleUV(vec2 uv, vec2 scalingVector){
    return uv/scalingVector;
}

float SDFU_ScaleUVRestore(float sdf, vec2 scalingVector){
    return sdf*length(scalingVector);
}

float SDFU_ScaleUVRestore2(float sdf, vec2 scalingVector){
    return sdf*max(scalingVector.x, scalingVector.y);
}



//! use unit size shapes, aka shapes that can live in a (0,1) space without feeling huge

float SDF_Circle(vec2 uv){
    float radius = 0.5;
    return length(uv) - radius;
}

float SDF_Band(vec2 uv){
    float halfSize = 0.5;
    return abs(uv.y) - halfSize;
}

float SDF_Ring(vec2 uv, float thickness){
    float radius = 0.5;
    float tube = 0.25*thickness;
    return abs(length(uv) - radius + tube) - tube;
}

float SDF_Rectangle(vec2 uv, float cornerRadius){
    vec2 hyperband = abs(uv) - 0.5 + cornerRadius;
    return length(max(hyperband, 0.0)) + min(max(hyperband.x, hyperband.y), 0.0) - cornerRadius;
}



float FX_GammaCorrection(float value){
    return pow(value, 1.0/2.2);
}


void main(){

    vec2 uv = (2.0 * gl_FragCoord.xy - u_resolution) / u_resolution.y;
    vec3 color = vec3(0.0);
    color = vec3(SDF_Circle(uv));
    color = vec3(SDFU_SoftEdge(SDF_Circle(uv), 0.01));

    //color = vec3(SDF_Band(uv));
    //color = vec3(SDF_Rectangle(uv, 0.0));
    //color = vec3(SDFU_SoftEdge(SDF_Rectangle(uv, 0.3), 0.3));
    //color = vec3(FX_GammaCorrection(SDFU_SoftEdge(SDF_Rectangle(uv, 0.0), 0.1)));

    //WARNING: This restoration of UV is not exact. Switch to a data driven model instead
    vec2 scalingVector = vec2(1.0, 0.5);
    vec2 uv0 = SDFU_ScaleUV(uv, scalingVector);
    float shape = SDF_Rectangle(uv, 0.1);
    shape = SDFU_ScaleUVRestore(shape, scalingVector);


    color = vec3(fract(shape));



    fragColor = vec4(color, 1.0);
}
