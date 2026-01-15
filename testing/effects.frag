#version 460
// Contrast GLSL Library - Apache 2.0 - https://www.apache.org/licenses/LICENSE-2.0

#ifdef GL
precision mediump float;
#endif

out vec4 fragColor;

uniform float u_time;
uniform vec2 u_resolution;
uniform vec2 u_mouse;

const float PI = 3.14159265359;



vec3 FX_Posterize(vec3 color, float samples){
    return floor(color*samples)/samples;
}

float FX_GammaCorrection(float value){
    return pow(value, 1.0/2.2);
}

float FX_Noise(vec2 uv, float seed, float density){
    vec2 p = floor(uv);
    p += seed;
    p = fract(p*vec2(123.34, 456.21));
    p += dot(p, p+45.32);
    return step(1.0 - density, fract(p.x*p.y));
}


float MATH_NormalDistribution(float value){
    return exp(-value*value) / sqrt(2*PI);
}

float MATH_Remap(float x, float lowIn, float highIn, float lowOut, float highOut){
    return (x - lowIn) / (highIn - lowIn) * (highOut - lowOut) + lowOut;
}

float MATH_Quantize(float value, float step){
    return floor(value * step)/step;
}

float MATH_Remap01(float x, float lowIn, float highIn){
    return (x - lowIn) / (highIn - lowIn);
}



float MASK_SoftEdge(float sdf, float blur){
    return smoothstep(0, -blur, sdf);
}

float MASK_HardEdge(float sdf){
    return step(0, -sdf);
}


vec2 SDFU_Pixelize(vec2 uv, float resolution){
    return floor(uv*resolution)/resolution;
}



float SuperCircle(vec2 uv, float n){
    return pow(pow(abs(uv.x), n) + pow(abs(uv.y), n), 1.0/n) - 0.5;
}

float SDF_Circle(vec2 uv){
    return length(uv) - 0.5;
}


void main(){
    vec2 uv = (2.0 * gl_FragCoord.xy - u_resolution) / u_resolution.y;
    vec3 color = vec3(0.0);

    color = FX_Posterize(vec3(uv.x), 10.0);
    float number = MATH_NormalDistribution(uv.x*2.0);
    color = vec3(MATH_Remap(number, 0.0, 0.4, 0.0, 1.0));
    color = vec3(MATH_Remap01(number, 0.0, 0.4));
    float circle = SDF_Circle(uv);
    float circle2 = SDF_Circle(uv+vec2(0.3, 0.0));


    float pixelDensity = 100.0;
    circle = SDF_Circle(SDFU_Pixelize(uv, pixelDensity));
    circle2 = SDF_Circle(SDFU_Pixelize(uv+vec2(0.5, 0.0), pixelDensity));
    //Takes the intersection of two SDFs
    float intersection12 = max(circle, circle2);


    circle = MASK_HardEdge(circle);
    circle = FX_Noise(uv*pixelDensity, u_time/400.0*circle, 0.5)*circle;

    circle2 = MASK_HardEdge(circle2);
    circle2 = FX_Noise(uv*pixelDensity, floor(u_time*6.0)*circle2/400.0, 0.5)*circle2;

    intersection12 = MASK_HardEdge(intersection12);
    intersection12 = FX_Noise(uv*pixelDensity, u_time/400.0, 0.5)*intersection12;


    circle = circle + circle2 - intersection12;
    circle = SuperCircle(uv, 0.4);



    circle = MASK_SoftEdge(circle, fwidth(circle)); color = vec3(circle);
    fragColor = vec4(color, 1.0);
}
