#version 460
// Contrast GLSL Library - Apache 2.0 - https://www.apache.org/licenses/LICENSE-2.0

#ifdef GL
precision mediump float;
#endif

out vec4 fragColor;

uniform float u_time;
uniform vec2 u_resolution;
uniform vec2 u_mouse;

mat2 Rotate2D(float angle){
    return mat2(cos(angle), -sin(angle),
                sin(angle), cos(angle));
}

float SDF2D_Circle(vec2 uv){
    return length(uv) - 0.5;
}

float MASK_HardEdge(float sdf){
    return step(0, -sdf);
}

float MASK_SoftEdge(float sdf, float blur){
    return smoothstep(0, -blur, sdf);
}

vec3 FX_ChromaticAberration(float amount, float softness, float mask, float sdf){
    //return vec3(smoothstep(softness, 0.0, sdf+amount), mask, smoothstep(softness, 0.0, sdf-amount));
    return vec3(smoothstep(softness, 0.0, sdf), mask, smoothstep(softness, 0.0, sdf));
}


void main(){
    vec2 uv = (2.0 * gl_FragCoord.xy - u_resolution) / u_resolution.y;
    vec3 color = vec3(0.0);

    float circle = SDF2D_Circle(uv);
    color = vec3(MASK_SoftEdge(circle, fwidth(circle)));
    color = FX_ChromaticAberration(0.11, 0.11, color.r, circle);


    fragColor = vec4(color, 1.0);
}
