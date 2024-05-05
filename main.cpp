#include <iostream>
#include <vector>

#include "raylib.h"

#include "sethread.h"
#include "mathpls.h"

#ifndef MODEL_FILE
#define MODEL_FILE "stanford-bunny.obj"
#endif

constexpr mathpls::vec3 g = {0, -50, 0};
constexpr float dt = .02f;

std::vector<mathpls::vec3> rest_vertices;
std::vector<mathpls::vec3> curr_vertices;
std::vector<mathpls::vec3> last_vertices;

mathpls::vec3 rest_center;
mathpls::vec3 curr_center;

Model model{};
Camera camera{};

#define NUM_THREAD 8
st::ThreadPool tp{NUM_THREAD};

void calcu_center() {
    curr_center = mathpls::vec3(0);
    for (auto& v : curr_vertices) {
        curr_center += v;
    }
    curr_center /= (float)curr_vertices.size();
}

void solve_constraints() {
    calcu_center();

    mathpls::mat3 H;
    for (int i = 0; i < curr_vertices.size(); i++) {
        auto qi = rest_vertices[i] - rest_center;
        auto pi = curr_vertices[i] - curr_center;
        H += mathpls::outerProduct(qi, pi);
    }

    auto [U, S, V] = mathpls::SVD(H);

    auto R = V * U.T();
    auto t = curr_center - R * rest_center;

    std::atomic_int remain = NUM_THREAD;
    tp.dispatch(curr_vertices.size(), [&](auto b, auto e) {
        for (int i = b; i < e; i++) {
            auto target = R * rest_vertices[i] + t;
            auto correct = (target - curr_vertices[i]) * 0.5;
            curr_vertices[i] += correct;
        }
        remain--;
    });

    while (remain > 0)
        std::this_thread::yield();
}

void collision_response() {
    tp.dispatch(curr_vertices.size(), [&](auto b, auto e) {
        for (int i = b; i < e; i++)
            if (curr_vertices[i].y < 0)
                curr_vertices[i].y = 0;
    });
}

void move_vertices() {
    tp.dispatch(curr_vertices.size(), [&](auto b, auto e) {
        for (int i = b; i < e; i++) {
            auto t = curr_vertices[i];
            curr_vertices[i] += t - last_vertices[i] + g*dt*dt;
            last_vertices[i] = t;
        }
    });
}

void step() {
    solve_constraints();
    collision_response();
    move_vertices();
}

void init_vertices() {
    rest_vertices.clear();
    curr_vertices.clear();
    last_vertices.clear();

    for (int i = 0; i < model.meshCount; i++) {
        auto& mesh = model.meshes[i];
        auto oz = curr_vertices.size();
        curr_vertices.resize(oz + mesh.vertexCount);
        last_vertices.resize(oz + mesh.vertexCount);
        rest_vertices.resize(oz + mesh.vertexCount);
        
        std::atomic_int remain = NUM_THREAD;
        tp.dispatch(mesh.vertexCount, [&](auto b, auto e) {
            for (size_t j = b; j < e; j++) {
                auto vertex = mesh.vertices + j * 3;
                last_vertices[oz + j] = rest_vertices[oz + j] = curr_vertices[oz + j] = {
                        vertex[0] * 32,
                        vertex[1] * 32 + 20,
                        vertex[2] * 32
                };
            }
            remain--;
        });
        while (remain > 0)
            std::this_thread::yield();
    }
    calcu_center();
    rest_center = curr_center;
}

void init() {
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(800, 500, "svd demo");
    SetTargetFPS(30);
    model = LoadModel(MODEL_FILE);

    // 初始化摄像机
    camera.position = { 0.0f, 25.0f, 30.0f }; //相机所在位置{x,y,z}
    camera.target = { 0.0f, 10.0f, 0.0f }; //相机朝向位置{x,y,z}
    camera.up = { 0.0f, 1.0f, 0.0f }; //相机正上方朝向矢量
    camera.fovy = 60.0f; //相机视野宽度
    camera.projection = CAMERA_PERSPECTIVE; //采用透视投影

    init_vertices();
}

void draw_particles() {
    for (auto& v : curr_vertices) {
        DrawPoint3D({v.x, v.y, v.z}, RED);
    }
}

void draw() {
    BeginDrawing();
    ClearBackground(WHITE);
    DrawFPS(0, 0);

    //以摄像机视角绘制3d内容
    BeginMode3D(camera);
    //绘制水平面网格
    DrawGrid(100, 5);
    //绘制Y轴
    DrawLine3D({0,100,0}, {0,-100,0}, BLACK);

    draw_particles();

    EndMode3D();
    EndDrawing();
}

int main() {
    init();

    bool pause = true;
    while (!WindowShouldClose()) {
        auto key = GetKeyPressed();
        if (key == KEY_SPACE)
            pause =!pause;
        if (key == KEY_R)
            init_vertices();
        if (!pause)
            step();
        draw();
    }

    CloseWindow();

    return 0;
}
