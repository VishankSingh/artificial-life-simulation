#include <vector>
#include <cmath>
#include <unistd.h>
#include <GLFW/glfw3.h>


#define WHITE_PARTICLES_NUM 1200
#define RED_PARTICLES_NUM 1200
#define YELLOW_PARTICLES_NUM 1200
//#define BLUE_PARTICLES_NUM 750

struct Particle {
    float x, y;  // Position
    float vx, vy; // Velocity
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
std::vector<Particle> blueParticles;

void UpdateParticles(std::vector<Particle>& particles1, std::vector<Particle>& particles2, float g) {
    for (auto& p1 : particles1) {
        float fx = 0, fy = 0;
        for (auto& p2 : particles2) {
            float dx = p1.x - p2.x;
            float dy = p1.y - p2.y;
            float d = sqrt(dx*dx + dy*dy);
            if (d > 0) {
                float F = g * 1/d;
                fx += (F * dx);
                fy += (F * dy);
            }
        }
        p1.vx += fx;
        p1.vy += fy;
        p1.vx /= 2;
        p1.vy /= 2;
        p1.x += p1.vx;
        p1.y += p1.vy;
        if (p1.x <= 0 || p1.x >= 1000) {p1.vx *= -10;}
        if (p1.y <= 0 || p1.y >= 1000) {p1.vy *= -10;}

    }
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
        for (float i = x - 1; i <= x + 1; ++i)
        {
            for (float j = y - 1; j <= y + 1; ++j) {
                glVertex2f(i, j);
            }
        }
    }

    glColor3f(red.r, red.g, red.b);
    for (const auto& p : red_particles_vector) {
        float x = p.x, y = p.y;
        for (float i = x - 1; i <= x + 1; ++i)
        {
            for (float j = y - 1; j <= y + 1; ++j) {
                glVertex2f(i, j);
            }
        }
    }

    glColor3f(yellow.r, yellow.g, yellow.b);
        for (const auto& p : yellow_particles_vector) {
        float x = p.x, y = p.y;
        for (float i = x - 1; i <= x + 1; ++i) {
            for (float j = y - 1; j <= y + 1; ++j) {
                glVertex2f(i, j);
            }
        }
    }

    glColor3f(blue.r, blue.g, blue.b);
    for (const auto& p : blue_particles_vector) {
        float x = p.x, y = p.y;
        for (float i = x - 1; i <= x + 1; ++i) {
            for (float j = y - 1; j <= y + 1; ++j) {
                glVertex2f(i, j);
            }
        }
    }

    glEnd();
}

void CreateParticles(std::vector<Particle>& particles_vector) {
    for (auto& p : particles_vector) {
        p.x = static_cast<float>(rand() % 1000); // Random initial position
        p.y = static_cast<float>(rand() % 1000);
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
    blueParticles.resize(BLUE_PARTICLES_NUM);
    CreateParticles(blueParticles);
#endif



    glfwInit();
    GLFWwindow* window = glfwCreateWindow(1000, 1000, "Particle Simulation", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // Set the clear color
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 1000, 0, 1000, -1, 1);


    // Main loop
    while (true) {
        UpdateParticles(WhiteParticles, WhiteParticles, -0.032);
        UpdateParticles(WhiteParticles, RedParticles, -0.017);
        UpdateParticles(WhiteParticles, YellowParticles, 0.034);

        UpdateParticles(RedParticles, RedParticles, -0.010);
        UpdateParticles(RedParticles, WhiteParticles, -0.034);

        UpdateParticles(YellowParticles, YellowParticles, 0.015);
        UpdateParticles(YellowParticles, WhiteParticles, -0.020);

        RenderParticles(WhiteParticles,
                        RedParticles,
                        YellowParticles,
                        blueParticles);
        glEnd();

        glFlush();
        glfwSwapBuffers(window);
        glfwPollEvents();
        usleep(20);
    }
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
