#include <iostream>
#include <fstream>
#include <vector>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <string>
#include <limits> // ��� ������ � ��������� min � max

enum LogLevel {
    DEBUG,
    INFO,
    ERROR
};

LogLevel currentLogLevel = DEBUG; // ������ ������� �����������

std::ofstream logFile;

// ������� �����������
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

void exampleFunction() {
    log(INFO, "������ �������.");
    log(DEBUG, "��������� ��������� ��� �������.");
    log(ERROR, "��������� �� ������.");
    log(INFO, "����� �������.");
}


// ��������� ��� �������� �������
struct Section {
    int Z = 0; // ���������� Z ��� ����� ������� (���������� �� ������ ������)
    std::vector<std::pair<int, int>> points; // ���������� X � Y ���� ����� � �������
};

// ��������� ��� ��������� �����
struct Header {
    uint16_t version;         // ������ �����
    int N;                    // ���������� �������
    uint16_t pointsPerSection; // ���������� ����� � �������
    uint16_t logIndex;        // ������ ������
    double datetime;          // ����/����� ���������
    uint16_t frontDiameter;   // ������� ��������� �����
    uint16_t middleDiameter;  // ������� ������� �����
    uint16_t backDiameter;    // ������� ������� �����
    uint16_t tipDiameter;     // ������� ��������
    uint16_t logLength;       // ����� ������
    uint8_t curvature;        // ��������
    int16_t curvatureDirection; // ����������� ��������
    int16_t taper;            // ����
    int16_t taperBase;        // ���� �����
    float physicalVolume;     // ���������� �����
    uint16_t flags;           // �����
    float encoderPulsePrice;  // ���� �������� ��������
};

// ���������� ���������� ��� ����������
float rotationX = 0.0f;
float rotationY = 0.0f;
float scale = 1.0f;
float cameraZ = -500.0f; // ������� ������ �� ��� Z

std::vector<Section> sections; // ��� �������
GLuint VAO, VBO; // ������ ��� �������� �����

// ������� ��� ������ ��������� �� �����
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
    file.seekg(1, std::ios::cur); // ������� ���������� �����
    file.read(reinterpret_cast<char*>(&header.curvatureDirection), sizeof(header.curvatureDirection));
    file.read(reinterpret_cast<char*>(&header.taper), sizeof(header.taper));
    file.read(reinterpret_cast<char*>(&header.taperBase), sizeof(header.taperBase));
    file.read(reinterpret_cast<char*>(&header.physicalVolume), sizeof(header.physicalVolume));
    file.read(reinterpret_cast<char*>(&header.flags), sizeof(header.flags));
    file.read(reinterpret_cast<char*>(&header.encoderPulsePrice), sizeof(header.encoderPulsePrice));

    // ���������� ��������� ����� � 0 �� 83
    file.seekg(83, std::ios::cur);

    return header;
}

// ������� ��� ������ ������� �� �����
std::vector<Section> readSections(std::ifstream& file, int N, std::streampos fileSize) {
    std::vector<Section> sections;
    for (int i = 0; i < N; ++i) {
        Section section;

        // ������ ���������� Z ��� ������� (integer)
        file.read(reinterpret_cast<char*>(&section.Z), sizeof(section.Z));
        log(INFO, "����� ������ Z ��� ������� " + std::to_string(i) + ": ������� � ����� = " + std::to_string(file.tellg()));

        // ����������� ��� �������� ���������� Z
        log(INFO, "������� " + std::to_string(i) + ": Z = " + std::to_string(section.Z));

        // ������ ���������� ����� (word)
        uint16_t M;
        file.read(reinterpret_cast<char*>(&M), sizeof(M));
        log(INFO, "����� ������ ���������� ����� ��� ������� " + std::to_string(i) + ": ������� � ����� = " + std::to_string(file.tellg()));

        // ����������� ���������� ����� ��� ��������
        log(INFO, "���������� ����� � ������� " + std::to_string(i) + ": " + std::to_string(M));

        // ������ ����� (X � Y � ������� smallint)
        for (int j = 0; j < M; ++j) {
            int16_t x, y;
            file.read(reinterpret_cast<char*>(&x), sizeof(x)); // ������ X
            file.read(reinterpret_cast<char*>(&y), sizeof(y)); // ������ Y
            section.points.emplace_back(x, y); // ���������� ����� � �������

            // ����������� �����
            log(INFO, "����� " + std::to_string(j) + ": X = " + std::to_string(x) + ", Y = " + std::to_string(y));

            // �������� ������� ����� ������ ������ �����
            log(INFO, "����� ������ ����� " + std::to_string(j) + " ��� ������� " + std::to_string(i) + ": ������� � ����� = " + std::to_string(file.tellg()));
        }

        sections.push_back(section); // ���������� ������� � ������

        // �������� ������ �� ������� �����
        if (file.tellg() >= fileSize) {
            log(ERROR, "���� �������� �� ����� �� ������� " + std::to_string(i));
            break;  // ����� �� �����, ���� ���� �������� �� �����
        }
    }
    return sections;
}


// �������
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

// ��������� ����� ��� ��������, ����������� � ���������������
void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
        rotationX += 0.005f;
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
        rotationX -= 0.005f;
    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
        rotationY += 0.005f;
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
        rotationY -= 0.005f;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        scale += 0.005f;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        scale -= 0.005f;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        cameraZ += 2.0f; // ������� ������ ������
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        cameraZ -= 2.0f; // ������� ������ �����
}

// ������� ��� ���������� ��������
GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    return shader;
}

// ������� ��� �������� ������� �� ������ ������� (�����)
int createBuffersFromSections() {
    std::vector<float> vertices;

    // ����������� ����� � ������ ������ ��� OpenGL (3D � Z ��� ����������� �������)
    for (const auto& section : sections) {
        size_t M = section.points.size();
        if (M == 0) continue; // ���������� ������ �������

        // ���������� ����� ������ ������� ������� � ���� ����������
        for (size_t i = 0; i < M; ++i) {
            // X, Y � ���������� �����, Z � ������������� ���������� ��� ����� �������
            vertices.push_back(static_cast<float>(section.points[i].first) / 10.0f);  // X
            vertices.push_back(static_cast<float>(section.points[i].second) / 10.0f); // Y
            vertices.push_back(static_cast<float>(section.Z) / 10.0f);                // Z

            // ��������� ������ ����� � ��������� (��� ������ ��� ��������� ����������)
            if (i + 1 < M) {
                vertices.push_back(static_cast<float>(section.points[i + 1].first) / 10.0f);
                vertices.push_back(static_cast<float>(section.points[i + 1].second) / 10.0f);
                vertices.push_back(static_cast<float>(section.Z) / 10.0f);
            }
            else {
                // ��������� ����������: ��������� ����� ����������� � ������
                vertices.push_back(static_cast<float>(section.points[0].first) / 10.0f);
                vertices.push_back(static_cast<float>(section.points[0].second) / 10.0f);
                vertices.push_back(static_cast<float>(section.Z) / 10.0f);
            }
        }
    }

    // ��������� ��������������� ����� ����� ���������
    for (size_t i = 0; i < sections.size() - 1; ++i) {
        const auto& section1 = sections[i];
        const auto& section2 = sections[i + 1];
        size_t M = std::min(section1.points.size(), section2.points.size());

        for (size_t j = 0; j < M; ++j) {
            // ��������� ����� ����� ��������� ���������
            vertices.push_back(static_cast<float>(section1.points[j].first) / 10.0f);
            vertices.push_back(static_cast<float>(section1.points[j].second) / 10.0f);
            vertices.push_back(static_cast<float>(section1.Z) / 10.0f);

            vertices.push_back(static_cast<float>(section2.points[j].first) / 10.0f);
            vertices.push_back(static_cast<float>(section2.points[j].second) / 10.0f);
            vertices.push_back(static_cast<float>(section2.Z) / 10.0f);
        }
    }

    // ��������� ������ � �����
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    log(INFO, "���������� ������ ��� ���������: " + std::to_string(vertices.size()));

    return vertices.size();

}



int main() {
    system("chcp 1251");

    logFile.open("log.txt");
    if (!logFile.is_open()) {
        std::cerr << "�� ������� ������� ���� ��� �����������." << std::endl;
        return -1;
    }

    // ������ ������ �����������
    log(INFO, "������ ���������.");

    // �������� ��������� �����
    std::ifstream file("model3.lprf", std::ios::binary);
    if (!file.is_open()) {
        log(ERROR, "�� ������� ������� ����.");
        return -1;
    }

    // ������ ���������
    Header header = readHeader(file);
    log(INFO, "���� ������: " + std::to_string(header.version));
    log(INFO, "���������� �������: " + std::to_string(header.N));

    // ������� � 130-�� ����� ����� ���������
    file.seekg(130, std::ios::beg);

    // �������� ������� �����
    file.seekg(0, std::ios::end);  // ������� � ����� �����, ����� ������ ��� ������
    std::streampos fileSize = file.tellg();  // ��������� ������� �����
    file.seekg(130, std::ios::beg);  // ������� ������� �� 130-� ����
    log(INFO, "������ �����: " + std::to_string(fileSize) + " ����");

    // ������ ������ �������
    sections = readSections(file, header.N, fileSize);
    file.close();

    // ������������� GLFW
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

    // ������������� GLEW
    if (glewInit() != GLEW_OK) {
        log(ERROR, "Failed to initialize GLEW");
        return -1;
    }

    // ���������� ��������
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);

    // �������� ��������� ���������
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    glUseProgram(shaderProgram);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // �������� ������� �� ������ ��������� ������
    int BZ = createBuffersFromSections();

    // �������� ���� ������� ��� ����������� ����������� 3D ��������
    glEnable(GL_DEPTH_TEST);

    // �������� ����
    while (!glfwWindowShouldClose(window)) {
        processInput(window);

        // ������� ������
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // �������
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::rotate(model, rotationX, glm::vec3(1.0f, 0.0f, 0.0f));
        model = glm::rotate(model, rotationY, glm::vec3(0.0f, 1.0f, 0.0f));
        model = glm::scale(model, glm::vec3(scale, scale, scale));

        glm::mat4 view = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, cameraZ)); // ������� ������ �� ��� Z

        glm::mat4 projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 1000.0f); // ��������

        GLuint modelLoc = glGetUniformLocation(shaderProgram, "model");
        GLuint viewLoc = glGetUniformLocation(shaderProgram, "view");
        GLuint projectionLoc = glGetUniformLocation(shaderProgram, "projection");

        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));

        // ��������� �����
        glBindVertexArray(VAO);
        glDrawArrays(GL_LINES, 0, BZ / 3); // ��� ��� � ��� 3 ���������� �� �������
        glBindVertexArray(0);

        // ��������� �����
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    // �������� ����� ����� ����������
    logFile.close();
    return 0;
}
