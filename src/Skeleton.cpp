//=============================================================================================
// Computer Graphics Sample Program: Ray-tracing-let
//=============================================================================================
#include "framework.h"

//http://paulbourke.net/geometry/circlesphere/
// levag
//pont feny forgat
//pofoz
//comment
float rnd() { return (float) rand() / RAND_MAX; }

const float epsilon = 0.0001f;
enum MaterialType {
    ROUGH, REFLECTIVE
};

struct Material {
    vec3 ka, kd, ks;
    float shininess;
    vec3 F0;
    MaterialType type;

    Material(MaterialType t) { type = t; }
};

struct RoughMaterial : Material {
    RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) : Material(ROUGH) {
        ka = _kd * M_PI;
        kd = _kd;
        ks = _ks;
        shininess = _shininess;
    }
};

vec3 operator/(vec3 num, vec3 denom) {
    return vec3(num.x / denom.x, num.y / denom.y, num.z / denom.z);
}

struct ReflectiveMaterial : Material {
    ReflectiveMaterial(vec3 n, vec3 kappa) : Material(REFLECTIVE) {
        vec3 one(1, 1, 1);
        F0 = ((n - one) * (n - one) + kappa * kappa) / ((n + one) * (n + one) + kappa * kappa);
    }
};

struct Hit {
    float t;
    vec3 position, normal;
    Material *material;

    Hit() { t = -1; }
};

struct Ray {
    vec3 start, dir;

    Ray(vec3 _start, vec3 _dir) {
        start = _start;
        dir = normalize(_dir);
    }
};

class Intersectable {
protected:
    Material *material;
public:
    virtual Hit intersect(const Ray &ray) = 0;
};

struct Chips : public Intersectable {
    vec3 center;
    float radius;

    Chips(const vec3 &_center, float _radius, Material *_material) {
        center = _center;
        radius = _radius;
        material = _material;
    }

    Hit intersect(const Ray &ray) {
        Hit hit;
        vec3 dist = ray.start - center;

        vec3 origin = ray.start;
        vec3 direction = ray.dir;


        float x = 1.3;
        float y = 1.2;
        float z = 1.1;

        float a = (x * direction.x * direction.x) + (y * direction.y * direction.y);
        float b = 2 * x * origin.x * direction.x + 2 * y * origin.y * direction.y - z * direction.z;
        float c = x * origin.x * origin.x + y * origin.y * origin.y - z * origin.z;


        float discr = b * b - 4.0f * a * c;
        if (discr < 0) return hit;
        float sqrt_discr = sqrtf(discr);
        float t1 = (-b + sqrt_discr) / 2.0f / a;    // t1 >= t2 for sure
        float t2 = (-b - sqrt_discr) / 2.0f / a;

        vec3 inter1, inter2; // holnap smukival xd



        if((inter1.x - center.x) * (inter1.x - center.x) + (inter1.y - center.y) * (inter1.y - center.y) + (inter1.z - center.z) * (inter1.z - center.z) > radius * radius)
            hit.t=t2;
        if((inter2.x - center.x) * (inter2.x - center.x) + (inter2.y - center.y) * (inter2.y - center.y) + (inter2.z - center.z) * (inter2.z - center.z) > radius * radius)
            hit.t=t1;
        if((inter2.x - center.x) * (inter2.x - center.x) + (inter2.y - center.y) * (inter2.y - center.y) + (inter2.z - center.z) * (inter2.z - center.z) <= radius * radius &&
           (inter1.x - center.x) * (inter1.x - center.x) + (inter1.y - center.y) * (inter1.y - center.y) + (inter1.z - center.z) * (inter1.z - center.z) <= radius * radius)
            hit.t = (t2 > 0) ? t2 : t1;
        if((inter2.x - center.x) * (inter2.x - center.x) + (inter2.y - center.y) * (inter2.y - center.y) + (inter2.z - center.z) * (inter2.z - center.z) > radius * radius &&
           (inter1.x - center.x) * (inter1.x - center.x) + (inter1.y - center.y) * (inter1.y - center.y) + (inter1.z - center.z) * (inter1.z - center.z) > radius * radius)
            hit.t=-1;


        hit.position = ray.start + ray.dir * hit.t;
        hit.normal = vec3(-2 * x * (hit.position.x) / z, 2 * y * (hit.position.y) / z, 1);
        hit.material = material;
        return hit;
    }
};

struct Dodekaeder : public Intersectable {
    vec3 center;
    std::vector<vec3> edge = {
            vec3(0, 0.618, 1.618), vec3(0, -0.618, 1.618), vec3(0, -0.618, -1.618), vec3(0, 0.618, -1.618),
            vec3(1.618, 0, 0.618), vec3(-1.618, 0, 0.618), vec3(-1.618, 0, -0.618), vec3(1.618, 0, -0.618),
            vec3(0.618, 1.618, 0), vec3(-0.618, 1.618, 0), vec3(-0.618, -1.618, 0), vec3(0.618, -1.618, 0),
            vec3(1, 1, 1), vec3(-1, 1, 1), vec3(-1, -1, 1), vec3(1, -1, 1), vec3(1, -1, -1), vec3(1, 1, -1),
            vec3(-1, 1, -1), vec3(-1, -1, -1)
    };

    std::vector<int> planes = {
            1, 2, 16, 5, 13,
            1, 13, 9, 10, 14,
            1, 14, 6, 15, 2,
            2, 15, 11, 12, 16,
            3, 4, 18, 8, 17,
            3, 17, 12, 11, 20,
            3, 20, 7, 19, 4,
            19, 10, 9, 18, 4,
            16, 12, 17, 8, 5,
            5, 8, 18, 9, 13,
            14, 10, 19, 7, 6,
            6, 7, 20, 11, 15
    };
    float radius;
    Material *keretMaterial;

    Dodekaeder(const vec3 &_center, float _radius, Material *_material, Material *_keretMaterial) {
        center = _center;
        radius = _radius;
        material = _material;
        keretMaterial = _keretMaterial;
    }

    bool RaySphere(vec3 p1, vec3 p2, vec3 sc) {
        float radius = 0.1f;
        float a, b, c;

        vec3 mainPoint;

        mainPoint.x = p2.x - p1.x;
        mainPoint.y = p2.y - p1.y;
        mainPoint.z = p2.z - p1.z;
        a = mainPoint.x * mainPoint.x + mainPoint.y * mainPoint.y + mainPoint.z * mainPoint.z;
        b = 2 * (mainPoint.x * (p1.x - sc.x) + mainPoint.y * (p1.y - sc.y) + mainPoint.z * (p1.z - sc.z));
        c = sc.x * sc.x + sc.y * sc.y + sc.z * sc.z;
        c += p1.x * p1.x + p1.y * p1.y + p1.z * p1.z;
        c -= 2 * (sc.x * p1.x + sc.y * p1.y + sc.z * p1.z);
        c -= radius * radius;
        float det = b * b - 4 * a * c;
        if (det < 0) {
            return false;
        }


        return true;
    }


    void getObjPlane(int i, float scale, vec3 *p, vec3 *normal) {
        vec3 p1 = edge[planes[5 * i] - 1], p2 = edge[planes[5 * i + 1] - 1], p3 = edge[planes[5 * i + 2] - 1];
        *normal = cross(p2 - p1, p3 - p1);
        if (dot(p1, *normal) < 0)
            *normal = -*normal; // normal vector should point outwards
        *p = p1 * scale + vec3(0, 0, 0.03f);
    }

    Hit intersect(const Ray &ray) {
        Hit hit;
        float scale = 1;
        for (int i = 0; i < 12; ++i) {
            vec3 p1, normal;
            getObjPlane(i, scale, &p1, &normal);
            float ti = abs(dot(normal, ray.dir)) > epsilon ? dot(p1 - ray.start, normal) / dot(normal, ray.dir) : -1;
            if (ti <= epsilon || (ti > hit.t && hit.t > 0))
                continue;
            vec3 pintersect = ray.start + ray.dir * ti;
            bool outside = false;
            for (int j = 0; j < 12; j++) { // check all other half spaces whether point is inside
                if (i == j)
                    continue;
                vec3 p11, n;
                getObjPlane(j, scale, &p11, &n);
                if (dot(n, pintersect - p11) > 0) {
                    outside = true;
                    break;
                }
            }
            if (!outside) {

                hit.t = ti;
                hit.position = pintersect;
                hit.normal = normalize(normal);
                hit.material = material;
                if (RaySphere(edge[planes[5 * i] - 1], edge[planes[5 * i + 1] - 1], hit.position) ||
                    RaySphere(edge[planes[5 * i + 1] - 1], edge[planes[5 * i + 2] - 1], hit.position) ||
                    RaySphere(edge[planes[5 * i + 2] - 1], edge[planes[5 * i + 3] - 1], hit.position) ||
                    RaySphere(edge[planes[5 * i + 3] - 1], edge[planes[5 * i + 4] - 1], hit.position) ||
                    RaySphere(edge[planes[5 * i + 4] - 1], edge[planes[5 * i] - 1], hit.position)) {
                    hit.material = keretMaterial;
                }

            }
        }
        return hit;
    }
};

class Camera {
    vec3 eye, lookat, right, up;
    float fov;
public:
    void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov) {
        eye = _eye;
        lookat = _lookat;
        fov = _fov;
        vec3 w = eye - lookat;
        float windowSize = length(w) * tanf(fov / 2);
        right = normalize(cross(vup, w)) * windowSize;
        up = normalize(cross(w, right)) * windowSize;
    }

    Ray getRay(int X, int Y) {
        vec3 dir =
                lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) -
                eye;
        return Ray(eye, dir);
    }

    void Animate(float dt) {
        vec3 d = eye - lookat;
        eye = vec3(d.x * cos(dt) + d.z * sin(dt), d.y, d.z * cos(dt) - d.x * sin(dt)) + lookat;
        set(eye, lookat, up, fov);
    }
};

struct Light {
    vec3 direction;
    vec3 Le;

    Light(vec3 _direction, vec3 _Le) {
        direction = normalize(_direction);
        Le = _Le;
    }
};


class Scene {
    std::vector<Intersectable *> objects;
    std::vector<Light *> lights;
    Camera camera;
    vec3 La;
public:
    void build() {
        vec3 eye = vec3(0, 0, 10), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
        float fov = 45 * M_PI / 180;
        camera.set(eye, lookat, vup, fov);

        La = vec3(0.4f, 0.4f, 0.4f);
        vec3 lightDirection(1, 1, 1), Le(2, 2, 2);
        lights.push_back(new Light(lightDirection, Le));

        vec3 kd1(0.3f, 0.2f, 0.1f), kd2(0.1f, 0.2f, 0.3f), ks(2, 2, 2);
        Material *fal = new RoughMaterial(kd1, ks, 50);

        vec3 n(0, 0, 0);
        vec3 kappa(1, 1, 1);
        Material *portal = new ReflectiveMaterial(n, kappa);//tökéletes tükör
        for (int i = 0; i < 1; i++) {
            // objects.push_back(new Dodekaeder(vec3(rnd() - 0.5f, rnd() - 0.5f, rnd() - 0.5f), rnd() * 0.1f, portal, fal));
            objects.push_back(new Chips(vec3(rnd() - 0.5f, rnd() - 0.5f, rnd() - 0.5f), rnd() * 0.1f, fal));
        }
    }

    void render(std::vector<vec4> &image) {
        long timeStart = glutGet(GLUT_ELAPSED_TIME);
        for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
            for (int X = 0; X < windowWidth; X++) {
                vec3 color = trace(camera.getRay(X, Y));
                image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
            }
        }
    }

    Hit firstIntersect(Ray ray) {
        Hit bestHit;
        for (Intersectable *object : objects) {
            Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
            if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t)) bestHit = hit;
        }
        if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
        return bestHit;
    }

    bool shadowIntersect(Ray ray) {    // for directional lights
        for (Intersectable *object : objects) if (object->intersect(ray).t > 0) return true;
        return false;
    }

    vec3 trace(Ray ray, int depth = 0) {
        if (depth > 5) return La;
        Hit hit = firstIntersect(ray);
        if (hit.t < 0) return La;


        vec3 outRadiance(0, 0, 0);
        if (hit.material->type == ROUGH) {
            outRadiance = hit.material->ka * La;
            for (Light *light : lights) {
                Ray shadowRay(hit.position + hit.normal * epsilon, light->direction);
                float cosTheta = dot(hit.normal, light->direction);
                if (cosTheta > 0 && !shadowIntersect(shadowRay)) {    // shadow computation
                    outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
                    vec3 halfway = normalize(-ray.dir + light->direction);
                    float cosDelta = dot(hit.normal, halfway);
                    if (cosDelta > 0)
                        outRadiance = outRadiance + light->Le * hit.material->ks *
                                                    powf(cosDelta, hit.material->shininess);
                }
            }

        }
        if (hit.material->type == REFLECTIVE) {
            vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
            float cosa = 1-dot(ray.dir, hit.normal);
            vec3 one(1, 1, 1);
            vec3 F = hit.material->F0 + (one - hit.material->F0) * pow(1 - cosa, 5);
            outRadiance = outRadiance + trace(Ray(hit.position + hit.normal * epsilon, reflectedDir), depth + 1) * F;
        }


        return outRadiance;
    }

    void Animate(float dt) { camera.Animate(dt); }
};

GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

// vertex shader in GLSL
const char *vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char *fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord);
	}
)";

class FullScreenTexturedQuad {
    unsigned int vao;    // vertex array object id and texture id
    unsigned int textureId = 0;
    Texture texture;

public:
    FullScreenTexturedQuad(int windowWidth, int windowHeight) {
        glGenVertexArrays(1, &vao);    // create 1 vertex array object
        glBindVertexArray(vao);        // make it active

        unsigned int vbo;        // vertex buffer objects
        glGenBuffers(1, &vbo);    // Generate 1 vertex buffer objects

        // vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
        glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
        float vertexCoords[] = {-1, -1, 1, -1, 1, 1, -1, 1};    // two triangles forming a quad
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords,
                     GL_STATIC_DRAW);       // copy to that part of the memory which is not modified
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

        glGenTextures(1, &textureId);
        glBindTexture(GL_TEXTURE_2D, textureId);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }

    void Loadtexture(std::vector<vec4> &image) {
        glBindTexture(GL_TEXTURE_2D, textureId);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowWidth, windowHeight, 0, GL_RGBA, GL_FLOAT, &image[0]);
    }

    void Draw() { // id emég kellhet
        glBindVertexArray(vao);    // make the vao and its vbos active playing the role of the data source
        int location = glGetUniformLocation(gpuProgram.getId(), "textureUnit");
        const unsigned int textureUnit = 0;
        if (location >= 0) {
            glUniform1i(location, textureUnit);
            glActiveTexture(GL_TEXTURE0 + textureUnit);
            glBindTexture(GL_TEXTURE_2D, textureId);
        }
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);    // draw two triangles forming a quad
    }
};

FullScreenTexturedQuad *fullScreenTexturedQuad;

// Initialization, create an OpenGL context
void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);
    scene.build();


    fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight);

    // create program for the GPU
    gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
    std::vector<vec4> image(windowWidth * windowHeight);
    scene.render(image);
    fullScreenTexturedQuad->Loadtexture(image);
    fullScreenTexturedQuad->Draw();
    glutSwapBuffers();                                    // exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
    scene.Animate(0.3f);
    glutPostRedisplay();
}