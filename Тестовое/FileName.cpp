#include <iostream>
#include <fstream>
#include <vector>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <string>
#include <limits>

// Logging setup
enum LogLevel {
    DEBUG,
    INFO,
    ERROR
};

LogLevel currentLogLevel = DEBUG;

std::ofstream logFile;

void log(LogLevel level, const std::string& message) {
    if (level >= currentLogLevel) {
        switch (level) {
        case DEBUG:
            logFile << "[DEBUG]: " << message << std::endl;
            break;
        case INFO:
            logFile << "[INFO]: " << message << std::endl;
            break;
        case ERROR:
            logFile << "[ERROR]: " << message << std::endl;
            break;
        }
    }
}

// Data structures
struct Section {
    int Z = 0;
    std::vector<std::pair<int, int>> points;
};

struct Header {
    uint16_t version;
    int N;
    uint16_t pointsPerSection;
    uint16_t logIndex;
    double datetime;
    uint16_t frontDiameter;
    uint16_t middleDiameter;
    uint16_t backDiameter;
    uint16_t tipDiameter;
    uint16_t logLength;
    uint8_t curvature;
    int16_t curvatureDirection;
    int16_t taper;
    int16_t taperBase;
    float physicalVolume;
    uint16_t flags;
    float encoderPulsePrice;
};

// Global variables for control
float rotationX = 0.0f;
float rotationY = 0.0f;
float scale = 1.0f;
float positionX = 0.0f;

std::vector<Section> sections;
GLuint VAO, VBO;

double lastX = 400, lastY = 300;
bool firstMouse = true;
bool mousePressed = false;
float sensitivity = 0.005f;

// Function to read header
Header readHeader(std::ifstream& file) {
    Header header;
    file.read(reinterpret_cast<char*>(&header.version), sizeof(header.version));
    file.read(reinterpret_cast<char*>(&header.N), sizeof(header.N));
    file.read(reinterpret_cast<char*>(&header.pointsPerSection), sizeof(header.pointsPerSection));
    file.read(reinterpret_cast<char*>(&header.logIndex), sizeof(header.logIndex));
    file.read(reinterpret_cast<char*>(&header.datetime), sizeof(header.datetime));
    file.read(reinterpret_cast<char*>(&header.frontDiameter), sizeof(header.frontDiameter));
    file.read(reinterpret_cast<char*>(&header.middleDiameter), sizeof(header.middleDiameter));
    file.read(reinterpret_cast<char*>(&header.backDiameter), sizeof(header.backDiameter));
    file.read(reinterpret_cast<char*>(&header.tipDiameter), sizeof(header.tipDiameter));
    file.read(reinterpret_cast<char*>(&header.logLength), sizeof(header.logLength));
    file.read(reinterpret_cast<char*>(&header.curvature), sizeof(header.curvature));
    file.seekg(1, std::ios::cur);
    file.read(reinterpret_cast<char*>(&header.curvatureDirection), sizeof(header.curvatureDirection));
    file.read(reinterpret_cast<char*>(&header.taper), sizeof(header.taper));
    file.read(reinterpret_cast<char*>(&header.taperBase), sizeof(header.taperBase));
    file.read(reinterpret_cast<char*>(&header.physicalVolume), sizeof(header.physicalVolume));
    file.read(reinterpret_cast<char*>(&header.flags), sizeof(header.flags));
    file.read(reinterpret_cast<char*>(&header.encoderPulsePrice), sizeof(header.encoderPulsePrice));
    file.seekg(83, std::ios::cur);
    return header;
}

// Function to read sections
std::vector<Section> readSections(std::ifstream& file, int N, std::streampos fileSize) {
    std::vector<Section> sections;
    for (int i = 0; i < N; ++i) {
        Section section;
        file.read(reinterpret_cast<char*>(&section.Z), sizeof(section.Z));
        log(INFO, "После чтения Z для сечения " + std::to_string(i) + ": позиция в файле = " + std::to_string(file.tellg()));
        log(INFO, "Сечение " + std::to_string(i) + ": Z = " + std::to_string(section.Z));
        uint16_t M;
        file.read(reinterpret_cast<char*>(&M), sizeof(M));
        log(INFO, "После чтения количества точек для сечения " + std::to_string(i) + ": позиция в файле = " + std::to_string(file.tellg()));
        log(INFO, "Количество точек в сечении " + std::to_string(i) + ": " + std::to_string(M));
        for (int j = 0; j < M; ++j) {
            int16_t x, y;
            file.read(reinterpret_cast<char*>(&x), sizeof(x));
            file.read(reinterpret_cast<char*>(&y), sizeof(y));
            section.points.emplace_back(x, y);
            log(INFO, "Точка " + std::to_string(j) + ": X = " + std::to_string(x) + ", Y = " + std::to_string(y));
            log(INFO, "После чтения точки " + std::to_string(j) + " для сечения " + std::to_string(i) + ": позиция в файле = " + std::to_string(file.tellg()));
        }
        sections.push_back(section);
        if (file.tellg() >= fileSize) {
            log(ERROR, "Файл прочитан до конца на сечении " + std::to_string(i));
            break;
        }
    }
    return sections;
}

// Shader sources
const char* vertexShaderSource = R"(
#version 330 core
layout(location = 0) in vec3 aPos;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
void main() {
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
)";

const char* fragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;
void main() {
    FragColor = vec4(1.0, 1.0, 1.0, 1.0);
}
)";

// Input processing
void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        positionX += 1.0f;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        positionX -= 1.0f;
}

// Mouse button callback
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            mousePressed = true;
            firstMouse = true;
        }
        else if (action == GLFW_RELEASE) {
            mousePressed = false;
        }
    }
}

// Mouse movement callback
void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    if (mousePressed) {
        if (firstMouse) {
            lastX = xpos;
            lastY = ypos;
            firstMouse = false;
        }
        float xoffset = xpos - lastX;
        float yoffset = ypos - lastY;
        lastX = xpos;
        lastY = ypos;
        rotationY += xoffset * sensitivity;
        rotationX += yoffset * sensitivity;
    }
}

// Scroll callback for zooming
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    scale += yoffset * 0.1f;
    if (scale < 0.1f)
        scale = 0.1f;
}

// Shader compilation
GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    return shader;
}

// Create buffers from sections and add end caps
int createBuffersFromSections() {
    std::vector<float> vertices;

    // Generate side triangles
    for (size_t i = 0; i < sections.size() - 1; ++i) {
        const auto& section1 = sections[i];
        const auto& section2 = sections[i + 1];
        size_t M1 = section1.points.size();
        size_t M2 = section2.points.size();
        size_t M = std::min(M1, M2);
        if (M < 3) continue;
        for (size_t j = 0; j < M; ++j) {
            size_t nextJ = (j + 1) % M;
            float x0 = static_cast<float>(section1.points[j].first) / 10.0f;
            float y0 = static_cast<float>(section1.points[j].second) / 10.0f;
            float z0 = static_cast<float>(section1.Z) / 10.0f;
            float x1 = static_cast<float>(section1.points[nextJ].first) / 10.0f;
            float y1 = static_cast<float>(section1.points[nextJ].second) / 10.0f;
            float z1 = static_cast<float>(section1.Z) / 10.0f;
            float x2 = static_cast<float>(section2.points[j].first) / 10.0f;
            float y2 = static_cast<float>(section2.points[j].second) / 10.0f;
            float z2 = static_cast<float>(section2.Z) / 10.0f;
            float x3 = static_cast<float>(section2.points[nextJ].first) / 10.0f;
            float y3 = static_cast<float>(section2.points[nextJ].second) / 10.0f;
            float z3 = static_cast<float>(section2.Z) / 10.0f;
            // First triangle
            vertices.push_back(x0); vertices.push_back(y0); vertices.push_back(z0);
            vertices.push_back(x1); vertices.push_back(y1); vertices.push_back(z1);
            vertices.push_back(x2); vertices.push_back(y2); vertices.push_back(z2);
            // Second triangle
            vertices.push_back(x1); vertices.push_back(y1); vertices.push_back(z1);
            vertices.push_back(x3); vertices.push_back(y3); vertices.push_back(z3);
            vertices.push_back(x2); vertices.push_back(y2); vertices.push_back(z2);
        }
    }

    // Add end cap for the first section
    {
        const auto& section = sections.front();
        size_t M = section.points.size();
        if (M >= 3) {
            float z0 = static_cast<float>(section.Z) / 10.0f;
            float centerX = 0.0f;
            float centerY = 0.0f;
            for (const auto& p : section.points) {
                centerX += static_cast<float>(p.first) / 10.0f;
                centerY += static_cast<float>(p.second) / 10.0f;
            }
            centerX /= M;
            centerY /= M;
            for (size_t j = 0; j < M; ++j) {
                size_t nextJ = (j + 1) % M;
                float x1 = static_cast<float>(section.points[j].first) / 10.0f;
                float y1 = static_cast<float>(section.points[j].second) / 10.0f;
                float x2 = static_cast<float>(section.points[nextJ].first) / 10.0f;
                float y2 = static_cast<float>(section.points[nextJ].second) / 10.0f;
                // Triangle
                vertices.push_back(centerX); vertices.push_back(centerY); vertices.push_back(z0);
                vertices.push_back(x1); vertices.push_back(y1); vertices.push_back(z0);
                vertices.push_back(x2); vertices.push_back(y2); vertices.push_back(z0);
            }
        }
    }

    // Add end cap for the last section
    {
        const auto& section = sections.back();
        size_t M = section.points.size();
        if (M >= 3) {
            float z0 = static_cast<float>(section.Z) / 10.0f;
            float centerX = 0.0f;
            float centerY = 0.0f;
            for (const auto& p : section.points) {
                centerX += static_cast<float>(p.first) / 10.0f;
                centerY += static_cast<float>(p.second) / 10.0f;
            }
            centerX /= M;
            centerY /= M;
            for (size_t j = 0; j < M; ++j) {
                size_t nextJ = (j + 1) % M;
                float x1 = static_cast<float>(section.points[j].first) / 10.0f;
                float y1 = static_cast<float>(section.points[j].second) / 10.0f;
                float x2 = static_cast<float>(section.points[nextJ].first) / 10.0f;
                float y2 = static_cast<float>(section.points[nextJ].second) / 10.0f;
                // Triangle
                vertices.push_back(centerX); vertices.push_back(centerY); vertices.push_back(z0);
                vertices.push_back(x2); vertices.push_back(y2); vertices.push_back(z0);
                vertices.push_back(x1); vertices.push_back(y1); vertices.push_back(z0);
            }
        }
    }

    // Upload data to buffers
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    log(INFO, "Количество вершин для отрисовки: " + std::to_string(vertices.size()));
    return vertices.size();
}

int main() {
    system("chcp 1251");
    logFile.open("log.txt");
    if (!logFile.is_open()) {
        std::cerr << "Не удалось открыть файл для логирования." << std::endl;
        return -1;
    }
    log(INFO, "Запуск программы.");
    std::ifstream file("model3.lprf", std::ios::binary);
    if (!file.is_open()) {
        log(ERROR, "Не удалось открыть файл.");
        return -1;
    }
    Header header = readHeader(file);
    log(INFO, "Файл версии: " + std::to_string(header.version));
    log(INFO, "Количество сечений: " + std::to_string(header.N));
    file.seekg(130, std::ios::beg);
    file.seekg(0, std::ios::end);
    std::streampos fileSize = file.tellg();
    file.seekg(130, std::ios::beg);
    log(INFO, "Размер файла: " + std::to_string(fileSize) + " байт");
    sections = readSections(file, header.N, fileSize);
    file.close();
    if (!glfwInit()) {
        log(ERROR, "Failed to initialize GLFW");
        return -1;
    }
    GLFWwindow* window = glfwCreateWindow(800, 600, "3D Object Viewer", nullptr, nullptr);
    if (!window) {
        log(ERROR, "Failed to create GLFW window");
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    if (glewInit() != GLEW_OK) {
        log(ERROR, "Failed to initialize GLEW");
        return -1;
    }
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetScrollCallback(window, scroll_callback);
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    glUseProgram(shaderProgram);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    int BZ = createBuffersFromSections();
    glEnable(GL_DEPTH_TEST);
    while (!glfwWindowShouldClose(window)) {
        processInput(window);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Set wireframe mode
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3(positionX, 0.0f, 0.0f));
        model = glm::rotate(model, rotationX, glm::vec3(1.0f, 0.0f, 0.0f));
        model = glm::rotate(model, rotationY, glm::vec3(0.0f, 1.0f, 0.0f));
        model = glm::scale(model, glm::vec3(scale, scale, scale));
        glm::mat4 view = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -500.0f));
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 1000.0f);
        GLuint modelLoc = glGetUniformLocation(shaderProgram, "model");
        GLuint viewLoc = glGetUniformLocation(shaderProgram, "view");
        GLuint projectionLoc = glGetUniformLocation(shaderProgram, "projection");
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));
        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, BZ / 3);
        glBindVertexArray(0);

        // Reset to fill mode if needed elsewhere
        // glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwTerminate();
    logFile.close();
    return 0;
}
