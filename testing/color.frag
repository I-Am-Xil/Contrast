#version 460
// Contrast GLSL Library - Apache 2.0 - https://www.apache.org/licenses/LICENSE-2.0

#ifdef GL
precision mediump float;
#endif

out vec4 fragColor;

uniform float u_time;
uniform vec2 u_resolution;
uniform vec2 u_mouse;

const float FLOAT_ERROR = 1e-4;
const float PI = 3.14159265359;

const struct TIME_stageConfig{
    float stageLength;
    int stages;
    int subdivisions;
    float loopLength;
    float stageSubdiv;
};

struct TIME_stageInfo{
    float loopTime;
    int currentStage;
    float stageProgress;
    float stageTime;
};

TIME_stageConfig TIME_SetStageConfig(float stageLength, int stages, int subdivisions){
    return TIME_stageConfig(stageLength, stages, subdivisions, stageLength*float(stages), stageLength/float(subdivisions));
}

TIME_stageInfo TIME_GetStageInfo(float time, TIME_stageConfig config){
    float loopTime = mod(time, config.loopLength);
    int currentStage = int(loopTime/config.stageLength);
    float stageProgress = fract(loopTime/config.stageLength);
    float stageTime = mod(loopTime, config.stageLength);
    return TIME_stageInfo(loopTime, currentStage, stageProgress, stageTime);
}


vec3 COLOR_NormalizeRGB(vec3 rgbColor){
    return rgbColor/255.0;
}

vec3 COLOR_HEXToRGB(int hexColor){
    float r = float((hexColor >> 16) & 0xFF)/255.0;
    float g = float((hexColor >> 8) & 0xFF)/255.0;
    float b = float(hexColor & 0xFF)/255.0;
    return vec3(r, g, b);
}

vec3 RGBToMyColorModel(vec3 rgbColor){
    vec3 U = normalize(vec3(1.0, -0.5, -0.5));
    vec3 V = normalize(vec3(0.0, 1.0, -1.0));
    float hue = atan(dot(normalize(rgbColor), V), dot(normalize(rgbColor), U));

    float saturation = 1.0 - dot(normalize(rgbColor), normalize(vec3(1.0)))/2.0;
    float brightness = length(rgbColor)/3.0;

    return vec3(hue, saturation, brightness);
}

vec3 RGBFromMyColorModel(vec3 MyColor){
    float hue = MyColor.x;
    float saturation = MyColor.y;
    float brightness = MyColor.z;

    vec3 U = normalize(vec3(1.0, -0.5, -0.5));
    vec3 V = normalize(vec3(0.0, 1.0, -1.0));

    vec3 hueVector = cos(hue)*V + sin(hue)*U;
    vec3 whiteDir = normalize(vec3(1.0));

    float scale = saturation / dot(hueVector, whiteDir);
    vec3 colorDir = hueVector*scale;

    vec3 rgbColor = normalize(colorDir)*brightness*3.0;

    return rgbColor;
}


void main(){
    vec2 uv = (2.0 * gl_FragCoord.xy - u_resolution) / u_resolution.y;
    vec3 color = vec3(0.0);

    color = COLOR_NormalizeRGB(vec3(255.0, 0.0, 0.0)) - vec3(1.0);
    color = COLOR_HEXToRGB(0x00F0F0);

    vec3 testColor = RGBToMyColorModel(vec3(1.0, 1.0, 1.0));

    color = RGBFromMyColorModel(testColor);
    //color = RGBFromMyColorModel(vec3(mod(u_time, PI), 1.0, 1.0));
    //color = RGBFromMyColorModel(vec3(2.0, mod(u_time, PI), 0.5));
    //color = RGBFromMyColorModel(vec3(0.0, 1.0, mod(u_time, PI)));


    fragColor = vec4(color, 1.0);
}
