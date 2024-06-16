#include <vector>
#include <GL/glut.h>
#include <unistd.h>

#define WHITE_PARTICLES_NUM 3000
#define RED_PARTICLES_NUM 3000
#define YELLOW_PARTICLES_NUM 3000
#define BLUE_PARTICLES_NUM 3000

#define REBOUND -5

struct Particle {
    float x, y;
    float vx, vy;
};

typedef struct Color {
    float r, g, b;
} Color;

Color white = {1, 1, 1};
Color red = {1, 0 , 0};
Color yellow = {1, 1 , 0};
Color blue = {0, 0 , 1};

std::vector<Particle> WhiteParticles;
std::vector<Particle> RedParticles;
std::vector<Particle> YellowParticles;
std::vector<Particle> BlueParticles;

__global__ void UpdateParticlesKernel(Particle* particles1, Particle* particles2, int numParticles1, int numParticles2, float g, float r) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numParticles1) {
        float fx = 0, fy = 0;
        for (int j = 0; j < numParticles2; ++j) {
            float dx = particles1[idx].x - particles2[j].x;
            float dy = particles1[idx].y - particles2[j].y;
            float d = sqrt(dx * dx + dy * dy);
            if (d > 0 && d < r) {
                float F = g / d;
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

void UpdateParticles(std::vector<Particle>& particles1, std::vector<Particle>& particles2, float g, float r) {
    int num_particles1 = particles1.size();
    int num_particles2 = particles2.size();
    Particle* d_particles1;
    Particle* d_particles2;
    size_t size1 = num_particles1 * sizeof(Particle);
    size_t size2 = num_particles2 * sizeof(Particle);

    cudaMalloc(&d_particles1, size1);
    cudaMalloc(&d_particles2, size2);

//    cudaError_t error = cudaGetLastError();
//    if (error != cudaSuccess) {
//        printf("CUDA error: %s\n", cudaGetErrorString(error));
//        exit(-1);
//    }

    cudaMemcpy(d_particles1, particles1.data(), size1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles2, particles2.data(), size2, cudaMemcpyHostToDevice);
    int threads_per_block = 1024;
    int blocks_per_grid = (num_particles1 + threads_per_block - 1) / threads_per_block;
    UpdateParticlesKernel<<<blocks_per_grid, threads_per_block>>>(d_particles1, d_particles2, num_particles1, num_particles2, g, r);
    cudaDeviceSynchronize();
    cudaMemcpy(particles1.data(), d_particles1, size1, cudaMemcpyDeviceToHost);
    cudaFree(d_particles1);
    cudaFree(d_particles2);
}

void RenderParticles(std::vector<Particle>& white_particles_vector,
                     std::vector<Particle>& red_particles_vector,
                     std::vector<Particle>& yellow_particles_vector,
                     std::vector<Particle>& blue_particles_vector) {
    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_POINTS);
    glColor3f(white.r, white.g, white.b);
    for (const auto& p : white_particles_vector) {
        float x = p.x, y = p.y;
        glVertex2f(x, y);
    }
    glColor3f(red.r, red.g, red.b);
    for (const auto& p : red_particles_vector) {
        float x = p.x, y = p.y;
        glVertex2f(x, y);
    }
    glColor3f(yellow.r, yellow.g, yellow.b);
    for (const auto& p : yellow_particles_vector) {
        float x = p.x, y = p.y;
        glVertex2f(x, y);
    }
    glColor3f(blue.r, blue.g, blue.b);
    for (const auto& p : blue_particles_vector) {
        float x = p.x, y = p.y;
        glVertex2f(x, y);
    }
    glEnd();
}

void CreateParticles(std::vector<Particle>& particles_vector) {
    for (auto& p : particles_vector) {
        p.x = static_cast<float>(rand() % 1080); // Random initial position
        p.y = static_cast<float>(rand() % 1920);
        p.vx = 0; // Random initial velocity
        p.vy = 0;
    }
}

int main(int argc, char** argv) {
#ifdef WHITE_PARTICLES_NUM
    WhiteParticles.resize(WHITE_PARTICLES_NUM);
    CreateParticles(WhiteParticles);
#endif

#ifdef RED_PARTICLES_NUM
    RedParticles.resize(RED_PARTICLES_NUM);
    CreateParticles(RedParticles);
#endif

#ifdef YELLOW_PARTICLES_NUM
    YellowParticles.resize(YELLOW_PARTICLES_NUM);
    CreateParticles(YellowParticles);
#endif

#ifdef BLUE_PARTICLES_NUM
    BlueParticles.resize(BLUE_PARTICLES_NUM);
    CreateParticles(BlueParticles);
#endif

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(1920, 1080);
    glutCreateWindow("Particle Simulation");
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 1920, 0, 1080, -1, 1);

    while (true) {
        RenderParticles(WhiteParticles,
                        RedParticles,
                        YellowParticles,
                        BlueParticles);

        UpdateParticles(YellowParticles, YellowParticles, -01.75, 51.65);
        UpdateParticles(YellowParticles, RedParticles, 0.499999, 116.575);
        UpdateParticles(YellowParticles, WhiteParticles, 0, 38.175);
        UpdateParticles(YellowParticles, BlueParticles, -01.1, 137.4);

        UpdateParticles(RedParticles, RedParticles, -01.5, 200);
        UpdateParticles(RedParticles, YellowParticles, 01.65, 108);
        UpdateParticles(RedParticles, WhiteParticles, 05.2, 79.825);
        UpdateParticles(RedParticles, BlueParticles, -10, 71.25);

        UpdateParticles(WhiteParticles, WhiteParticles, -01.2, 120.25);
        UpdateParticles(WhiteParticles, RedParticles, -08.4, 57.775);
        UpdateParticles(WhiteParticles, YellowParticles, 0.95, 144.75);
        UpdateParticles(WhiteParticles, BlueParticles, 0.3, 57.775);

        UpdateParticles(BlueParticles, BlueParticles, -01.2, 55.325);
        UpdateParticles(BlueParticles, WhiteParticles, -0.6, 152.1);
        UpdateParticles(BlueParticles, RedParticles, 06.65, 63.9);
        UpdateParticles(BlueParticles, YellowParticles, -0.65, 116.675);

        glFlush();
        usleep(1500);
    }
    return 0;
}
