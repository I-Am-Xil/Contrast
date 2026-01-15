#version 460
// Contrast GLSL Library - Apache 2.0 - https://www.apache.org/licenses/LICENSE-2.0

#ifdef GL
precision mediump float;
#endif

out vec4 fragColor;

uniform float u_time;
uniform vec2 u_resolution;
uniform vec2 u_mouse;

float Polygon(vec2 uv, float radius, float sides, float blur){
    float angle = atan(uv.x, uv.y);
    float slice = PI * 2.0 / sides;
    return smoothstep(radius,  radius - blur, cos(floor(0.5 + angle / slice) * slice - angle) * length(uv));
}

float PolygonSDF(vec2 uv, float radius, float sides){
    float UVangle = atan(uv.x, uv.y);
    float UVradius = length(uv);

    float slice = 2.0 * PI / sides;
    float height = cos(PI/sides);

    float localAngle = mod(UVangle + PI / sides, slice) - slice/2.0;

    float distance = cos(localAngle)*UVradius;

    return distance - radius*height;
}

float SmoothPolygonSDF(vec2 uv, float radius, float sides, float blur){
    return smoothstep(blur, -blur, PolygonSDF(uv, radius, sides));
}


void main(){
    vec2 uv = (2.0 * gl_FragCoord.xy - u_resolution) / u_resolution.y;
    vec3 color = vec3(0.0);

    color = vec3(Polygon(uv, 0.2, 6, 0.05));
    color = vec3(SmoothPolygonSDF(uv, 0.3, 6, 0.01));

    fragColor = vec4(color, 1.0);
}
