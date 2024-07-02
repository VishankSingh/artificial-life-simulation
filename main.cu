#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <GLFW/glfw3.h>
#include <vector>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <random>

#define PARTICLES_1_NUM 1000
#define PARTICLES_2_NUM 1000
#define PARTICLES_3_NUM 1000
#define PARTICLES_4_NUM 1000

constexpr signed int REBOUND = -5;

constexpr uint WINDOW_WIDTH = 1920;
constexpr uint WINDOW_HEIGHT = 1080;

constexpr uint ROWS = 4;
constexpr uint COLS = 4;

constexpr uint FRAME_DELAY_MICROSECONDS = 15000;

struct Particle {
    float x, y;
    float vx, vy;
};

struct Color {
    float r, g, b;
};

Color color_1 = {1, 1, 1};
Color color_2 = {1, 0, 0};
Color color_3 = {1, 1, 0};
Color color_4 = {0, 0, 1};

std::vector<Particle> ParticlesType1;
std::vector<Particle> ParticlesType2;
std::vector<Particle> ParticlesType3;
std::vector<Particle> ParticlesType4;

bool ReadFile(const std::string &filename, double (&force)[ROWS][COLS]) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[-] Unable to open file " << filename << std::endl;
        return false;
    }

    std::string line;
    int row = 0;

    while (std::getline(file, line) && row < ROWS) {
        std::istringstream iss(line);
        std::string token;
        int col = 0;

        while (std::getline(iss, token, ',') && col < COLS) {
            try {
                force[row][col] = std::stod(token);
            } catch (const std::invalid_argument& e) {
                std::cerr << "[-] Invalid argument: " << e.what() << " at row " << row << ", col " << col << std::endl;
                return false;
            } catch (const std::out_of_range& e) {
                std::cerr << "[-] Out of range: " << e.what() << " at row " << row << ", col " << col << std::endl;
                return false;
            }
            ++col;
        }

        if (col != COLS) {
            std::cerr << "[-] Incorrect number of columns in line " << row + 1 << std::endl;
            file.close();
            return false;
        }
        ++row;
    }
    if (row != ROWS) {
        std::cerr << "[-] Incorrect number of rows in file " << filename << std::endl;
        return false;
    }
    return true;
}

void CreateParticles(std::vector<Particle> &particles_vector) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist_x(0.0f, 1920.0f);
    std::uniform_real_distribution<float> dist_y(0.0f, 1080.0f);

    for (auto &p : particles_vector) {
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.vx = 0.0f;
        p.vy = 0.0f;
    }
}

__global__ void UpdateParticlesKernel(Particle * __restrict__ particles1,
                                      const Particle * __restrict__ particles2,
                                      int numParticles1,
                                      int numParticles2,
                                      float force,
                                      float dist) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < numParticles1; i += stride) {
        float fx = 0.0f, fy = 0.0f;
        for (int j = 0; j < numParticles2; ++j) {
            float dx = particles1[i].x - particles2[j].x;
            float dy = particles1[i].y - particles2[j].y;
            float d = sqrtf(dx * dx + dy * dy);
            if (0.0f < d && d < dist) {
                float F = force / d;
                fx += (F * dx);
                fy += (F * dy);
            }
        }
        particles1[idx].vx = (particles1[idx].vx + fx) / 2;
        particles1[idx].vy = (particles1[idx].vy + fy) / 2;
        particles1[idx].x += particles1[idx].vx;
        particles1[idx].y += particles1[idx].vy;
        if (particles1[idx].x <= 0 || particles1[idx].x >= 1920) { particles1[idx].vx *= REBOUND; }
        if (particles1[idx].y <= 0 || particles1[idx].y >= 1080) { particles1[idx].vy *= REBOUND; }
    }
}

void UpdateParticles(std::vector<Particle> &particles1, std::vector<Particle> &particles2, float force, float dist) {
    int num_particles1 = particles1.size();
    int num_particles2 = particles2.size();
    Particle *d_particles1;
    Particle *d_particles2;

    size_t size1 = num_particles1 * sizeof(Particle);
    size_t size2 = num_particles2 * sizeof(Particle);

    cudaMalloc(&d_particles1, size1);
    cudaMalloc(&d_particles2, size2);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("[-] CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaMemcpy(d_particles1, particles1.data(), size1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles2, particles2.data(), size2, cudaMemcpyHostToDevice);

    int threads_per_block = 1024;
    int blocks_per_grid = (num_particles1 + threads_per_block - 1) / threads_per_block;
    UpdateParticlesKernel<<<blocks_per_grid, threads_per_block>>>(d_particles1,
                                                                  d_particles2,
                                                                  num_particles1,
                                                                  num_particles2,
                                                                  force,
                                                                  dist);
    cudaDeviceSynchronize();
    cudaMemcpy(particles1.data(), d_particles1, size1, cudaMemcpyDeviceToHost);
    cudaFree(d_particles1);
    cudaFree(d_particles2);
}

void RenderParticles(const std::vector<Particle> &particles_1_vector,
                     const std::vector<Particle> &particles_2_vector,
                     const std::vector<Particle> &particles_3_vector,
                     const std::vector<Particle> &particles_4_vector) {
    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_POINTS);

    auto renderParticleVector = [](const std::vector<Particle> &particles, const Color &color) {
        glColor3f(color.r, color.g, color.b);
        for (const auto &p : particles) {
            glVertex2f(p.x, p.y);
        }
    };

#ifdef PARTICLES_1_NUM
    renderParticleVector(particles_1_vector, color_1);
#endif

#ifdef PARTICLES_2_NUM
    renderParticleVector(particles_2_vector, color_2);
#endif

#ifdef PARTICLES_3_NUM
    renderParticleVector(particles_3_vector, color_3);
#endif

#ifdef PARTICLES_4_NUM
    renderParticleVector(particles_4_vector, color_4);
#endif

    glEnd();
}

int main(int argc, char **argv) {
#ifdef PARTICLES_1_NUM
    ParticlesType1.resize(PARTICLES_1_NUM);
    CreateParticles(ParticlesType1);
#endif

#ifdef PARTICLES_2_NUM
    ParticlesType2.resize(PARTICLES_2_NUM);
    CreateParticles(ParticlesType2);
#endif

#ifdef PARTICLES_3_NUM
    ParticlesType3.resize(PARTICLES_3_NUM);
    CreateParticles(ParticlesType3);
#endif

#ifdef PARTICLES_4_NUM
    ParticlesType4.resize(PARTICLES_4_NUM);
    CreateParticles(ParticlesType4);
#endif

    glfwInit();
    if (!glfwInit()) {
        std::cerr << "[-] Failed to initialize GLFW" << std::endl;
        return -1;
    }

    GLFWwindow *window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Artificial Constructs", NULL, NULL);
    if (!window) {
        std::cerr << "[-] Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, WINDOW_WIDTH, 0, WINDOW_HEIGHT, -1, 1);

    double force[ROWS][COLS];
    double dist[ROWS][COLS];

    ReadFile("force.txt", force);
    ReadFile("distance.txt", dist);

    while (!glfwWindowShouldClose(window)) {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }

        RenderParticles(ParticlesType1,
                        ParticlesType2,
                        ParticlesType3,
                        ParticlesType4);

        UpdateParticles(ParticlesType1, ParticlesType1, force[0][0], dist[0][0]);
        UpdateParticles(ParticlesType1, ParticlesType2, force[0][1], dist[0][1]);
        UpdateParticles(ParticlesType1, ParticlesType3, force[0][2], dist[0][2]);
        UpdateParticles(ParticlesType1, ParticlesType4, force[0][3], dist[0][3]);

        UpdateParticles(ParticlesType2, ParticlesType1, force[1][0], dist[1][0]);
        UpdateParticles(ParticlesType2, ParticlesType2, force[1][1], dist[1][1]);
        UpdateParticles(ParticlesType2, ParticlesType3, force[1][2], dist[1][2]);
        UpdateParticles(ParticlesType2, ParticlesType4, force[1][3], dist[1][3]);

        UpdateParticles(ParticlesType3, ParticlesType1, force[2][0], dist[2][0]);
        UpdateParticles(ParticlesType3, ParticlesType2, force[2][1], dist[2][1]);
        UpdateParticles(ParticlesType3, ParticlesType3, force[2][2], dist[2][2]);
        UpdateParticles(ParticlesType3, ParticlesType4, force[2][3], dist[2][3]);

        UpdateParticles(ParticlesType4, ParticlesType1, force[3][0], dist[3][0]);
        UpdateParticles(ParticlesType4, ParticlesType2, force[3][1], dist[3][1]);
        UpdateParticles(ParticlesType4, ParticlesType3, force[3][2], dist[3][2]);
        UpdateParticles(ParticlesType4, ParticlesType4, force[3][3], dist[3][3]);

        glFlush();
        glfwSwapBuffers(window);
        glfwPollEvents();
        usleep(FRAME_DELAY_MICROSECONDS);
    }
    glfwTerminate();
    return 0;
}
