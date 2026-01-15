#version 460
// Contrast GLSL Library - Apache 2.0 - https://www.apache.org/licenses/LICENSE-2.0

#ifdef GL
precision mediump float;
#endif

out vec4 fragColor;

uniform float u_time;
uniform vec2 u_resolution;
uniform vec2 u_mouse;

float SDFCircle(vec2 uv, vec2 position, float radius){
    float d = length(uv - position);
    return d - radius;
}

float SDFCircunference(vec2 uv, vec2 position, float radius, float thickness){
    float circ1 = SDFCircle(uv, position, radius + thickness/2.);
    float circ2 = SDFCircle(uv, position, radius - thickness/2.);
    return max(circ1, -circ2);
}

vec2 Perpendicular(vec2 direction){
    return vec2(-direction.y, direction.x);
}

float SDFBand(vec2 uv, float position, float thickness, vec2 direction){
    float halfThickness = thickness*0.5;
    float axis = dot(Perpendicular(normalize(direction)), uv) + position;
    return abs(axis) - halfThickness;
}

float SoftEdge(vec2 uv, float sdf, float blur){
    return smoothstep(0, -blur, sdf);
}

float Plot(vec2 coord, float plot){
    return smoothstep(plot - 0.02, plot, coord.y) - smoothstep(plot, plot + 0.02, coord.y);
}

void main(){
    vec2 uv = (2.0 * gl_FragCoord.xy - u_resolution) / u_resolution.y;
    vec3 color = vec3(0.0);
    color = vec3(SDFCircle(uv, vec2(0.0), 0.5));
    color = vec3(SoftEdge(uv, SDFBand(uv, 0.0, 0.5, vec2(1.0, 0.0)), 0.05) );

    color = vec3(SDFCircunference(uv, vec2(0.0), 0.5, 1.0));
    color = vec3(SoftEdge(uv, SDFCircunference(uv, vec2(0.0), 0.5, 0.1), 0.05));

    fragColor = vec4(color, 1.0);
}
