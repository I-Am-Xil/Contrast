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

float Circle(vec2 uv, float scale, float blur){
    return smoothstep(scale + blur, scale - blur, length(uv));
}

float Circunference(vec2 uv, float scale, float thickness, float blur){
    float circunference = Circle(uv, scale + thickness/2.0, blur);
    circunference -= Circle(uv, scale - thickness/2.0, blur);
    return circunference;
}

float BandOrthogonal(vec2 uv, float position, float thickness, vec2 blur, vec2 direction){
    float halfThickness = thickness*0.5;
    float leftEdge = position + halfThickness;
    float rightEdge = position - halfThickness;
    float axis = dot(normalize(direction), uv);

    float left = smoothstep(leftEdge + blur.x, leftEdge - blur.x, axis);
    float right = smoothstep(rightEdge + blur.y, rightEdge - blur.y, axis);
    return left - right;
}

vec2 Perpendicular(vec2 direction){
    return vec2(-direction.y, direction.x);
}

float BandAligned(vec2 uv, float position, float thickness, vec2 blur, vec2 direction){
    return BandOrthogonal(uv, position, thickness, blur, Perpendicular(direction));
}

float Rectangle(vec2 uv, vec2 position, vec2 size, vec4 blur, vec2 orientation){
    float xBand = BandOrthogonal(uv, position.x, size.x, blur.xy, orientation);
    float yBand = BandAligned(uv, position.y, size.y, blur.zw, orientation);
    return xBand * yBand;
}

float RectanglePerimeter(vec2 uv, vec2 position, vec2 size, float thickness, vec4 blur, vec2 orientation){
    float rect = Rectangle(uv, position, size + vec2(thickness), blur, orientation);
    rect -= Rectangle(uv, position, size - vec2(thickness), blur, orientation);
    return rect;
}

mat2 Rotate2D(float angle){
    return mat2(cos(angle), -sin(angle),
                sin(angle), cos(angle));
}

float Arc(vec2 uv, vec2 position, float radius, float angle, float thickness, vec4 blur){
    vec2 UVlocal = Rotate2D(angle/2.0)*uv + position;
    vec2 orientation = vec2(1.0, 0.0);

    float UVradius = length(UVlocal);
    float UVangle = atan(UVlocal.y, UVlocal.x);
    vec2 UVpolar = vec2(UVradius, UVangle);

    float arc = Rectangle(UVpolar, vec2(0.0), vec2(radius + thickness, angle), blur, orientation);
    arc -= Rectangle(UVpolar, vec2(0.0), vec2(radius - thickness, angle), blur, orientation);

    return arc;
}


void main(){
    vec2 uv = (2.0 * gl_FragCoord.xy - u_resolution) / u_resolution.y;
    vec3 color = vec3(0.0);

    color = vec3(Circunference(uv, 0.5, 0.1, 0.05));
    color = vec3(BandAligned(uv, 0.0, 0.5, vec2(0.05), vec2(1.0, 0.0)));
    color = vec3(Rectangle(uv, vec2(0.2, 0.0), vec2(0.5, 0.2), vec4(vec2(0.05), vec2(0.01)), vec2(1.0, 0.0)));
    color = vec3(Arc(uv, vec2(0.0), 1.0, 2.14, 0.1, vec4(0.05)));

    fragColor = vec4(color, 1.0);
}
