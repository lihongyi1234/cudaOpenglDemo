#include "cuda_opengl_demo.h"
#include "shader_tools/GLSLProgram.h"
#include "shader_tools/GLSLShader.h"
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include <cuda_texture_types.h>
#include <cuda_gl_interop.h>
#include <iostream>
#pragma comment(lib, "glfw3.lib")
#pragma comment(lib, "cudart.lib")

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

const char* vertexShaderSource = "#version 330 core\n"
"layout (location = 0) in vec3 aPos;\n"
"layout (location = 1) in vec3 aColor;\n"
"layout (location = 2) in vec2 aTexCoord;\n"
"out vec2 TexCoord;\n"
"out vec3 ourColor;\n"
"void main()\n"
"{\n"
"   gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
"   ourColor = aColor;\n"
"   TexCoord = aTexCoord;\n"
"}\0";
const char* fragmentShaderSource = "#version 330 core\n"
"out vec4 FragColor;\n"
"in vec3 ourColor;\n"
"in vec2 TexCoord;\n"
"uniform sampler2D ourTexture;\n"
"void main()\n"
"{\n"
"   FragColor = texture(ourTexture,TexCoord);\n"
"}\n\0";

GLFWwindow* window = nullptr;


int initOpengl()
{
	// glfw: initialize and configure
	// ------------------------------
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

	// glfw window creation
	// --------------------
	window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

	// glad: load all OpenGL function pointers
	// ---------------------------------------
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}
	return 0;
}

void uploadMat8UC4(const cv::Mat& image, cudaArray_t& cuArray, cudaTextureObject_t& texture)
{
	//The texture description
	cudaTextureDesc uchar1_texture_desc;
	memset(&uchar1_texture_desc, 0, sizeof(uchar1_texture_desc));
	uchar1_texture_desc.addressMode[0] = cudaAddressModeClamp;
	uchar1_texture_desc.addressMode[1] = cudaAddressModeClamp;
	uchar1_texture_desc.addressMode[2] = cudaAddressModeClamp;
	uchar1_texture_desc.filterMode = cudaFilterModePoint;
	uchar1_texture_desc.readMode = cudaReadModeElementType;
	uchar1_texture_desc.normalizedCoords = 0;

	//Create channel descriptions
	cudaChannelFormatDesc uchar1_channel_desc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);

	//Create the resource desc
	cudaResourceDesc resource_desc;
	//cudaArray* _cuArray;
	cudaMallocArray(&cuArray, &uchar1_channel_desc, image.cols, image.rows);
	memset(&resource_desc, 0, sizeof(cudaResourceDesc));
	resource_desc.resType = cudaResourceTypeArray;
	resource_desc.res.array.array = cuArray;

	//Allocate the texture
	//cudaTextureObject_t _texture = 0;
	cudaCreateTextureObject(&texture, &resource_desc, &uchar1_texture_desc, 0);
	cudaMemcpyToArray(cuArray, 0, 0, image.data, sizeof(uchar4) * image.rows * image.cols, cudaMemcpyHostToDevice);
}

// QUAD GEOMETRY
GLfloat vertices[] = {
	// Positions          // Colors           // Texture Coords
	1.0f, 1.0f, 0.5f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f,  // Top Right
	1.0f, -1.0f, 0.5f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,  // Bottom Right
	-1.0f, -1.0f, 0.5f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,  // Bottom Left
	-1.0f, 1.0f, 0.5f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f // Top Left 
};
// you can also put positions, colors and coordinates in seperate VBO's
GLuint indices[] = {  // Note that we start from 0!
	0, 1, 3,  // First Triangle
	1, 2, 3   // Second Triangle
};

int main(int argc, char** argv)
{
	if (initOpengl() < 0) {
		return -1;
	}
	std::string jpg_fn = argv[1];

	int width = 2048;
	int height = 2048;

	cv::Mat atlas = cv::imread(jpg_fn);
	cv::resize(atlas, atlas, cv::Size(width, height));
	//upload texture to gpu
	//upload cv::Mat uchar to device 

	cudaArray_t cuArray;
	cudaTextureObject_t cuTexture;
	cv::Mat src4;
	cv::cvtColor(atlas, src4, CV_BGR2RGBA);
	uploadMat8UC4(src4, cuArray, cuTexture);

	unsigned int* cuda_dest_buffer;
	size_t size_tex_data = width * height * sizeof(uchar4);
	cudaMalloc((void**)&cuda_dest_buffer, size_tex_data);

	GLSLShader drawtex_v = GLSLShader("Textured draw vertex shader", vertexShaderSource, GL_VERTEX_SHADER);
	GLSLShader drawtex_f = GLSLShader("Textured draw fragment shader", fragmentShaderSource, GL_FRAGMENT_SHADER);
	GLSLProgram shdrawtex = GLSLProgram(&drawtex_v, &drawtex_f);
	shdrawtex.compile();

	unsigned int VBO, VAO, EBO;
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &EBO);
	// bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	// Position attribute (3 floats)
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (GLvoid*)0);
	glEnableVertexAttribArray(0);
	// Color attribute (3 floats)
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
	glEnableVertexAttribArray(1);
	// Texture attribute (2 floats)
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (GLvoid*)(6 * sizeof(GLfloat)));
	glEnableVertexAttribArray(2);

	// note that this is allowed, the call to glVertexAttribPointer registered VBO as the vertex attribute's bound vertex buffer object so afterwards we can safely unbind
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// remember: do NOT unbind the EBO while a VAO is active as the bound element buffer object IS stored in the VAO; keep the EBO bound.
	//glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	// You can unbind the VAO afterwards so other VAO calls won't accidentally modify this VAO, but this rarely happens. Modifying other
	// VAOs requires a call to glBindVertexArray anyways so we generally don't unbind VAOs (nor VBOs) when it's not directly necessary.
	glBindVertexArray(0);

	//texture
	cudaGraphicsResource_t cuda_tex_resource;
	GLuint opengl_tex_cuda;  // OpenGL Texture for cuda result
	glGenTextures(1, &opengl_tex_cuda);
	glBindTexture(GL_TEXTURE_2D, opengl_tex_cuda);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glGenerateMipmap(GL_TEXTURE_2D);
	//在CUDA中注册这个Texture
	cudaError_t	err = cudaGraphicsGLRegisterImage(&cuda_tex_resource, opengl_tex_cuda, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
	if (err != cudaSuccess) {
		std::cout << "cudaGraphicsGLRegisterImage: " << err << "line: " << __LINE__;
		return -1;
	}
	// 在CUDA中锁定资源，获得操作Texture的指针，这里是CudaArray*类型
	//launch_cudaProcess(cuda_dest_buffer, width, height);
	cudaArray_t texture_ptr;
	err = cudaGraphicsMapResources(1, &cuda_tex_resource);
	err = cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_tex_resource, 0, 0);
	//cudaMemcpyToArray(texture_ptr, 0, 0, cuArray, size_tex_data, cudaMemcpyDeviceToDevice);
	cudaMemcpy2DArrayToArray(texture_ptr, 0, 0,
		cuArray, 0, 0, width * sizeof(uchar4), height, cudaMemcpyDeviceToDevice);
	// 处理完后即解除资源锁定，OpenGL可以利用得到的Texture对象进行纹理贴图操作了。
	cudaGraphicsUnmapResources(1, &cuda_tex_resource);

	// render loop
	// -----------
	while (!glfwWindowShouldClose(window))
	{
		// input
		// -----
		processInput(window);
		// render
		// ------
		glClearColor(1.f, 1.f, 1.f, 0.f);
		//glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
		glActiveTexture(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, opengl_tex_cuda);

		// draw our first triangle
		//glUseProgram(shaderProgram);
		shdrawtex.use();

		glBindVertexArray(VAO); // seeing as we only have a single VAO there's no need to bind it every time, but we'll do so to keep things a bit more organized
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
		// glBindVertexArray(0); // no need to unbind it every time 

		// glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
		// -------------------------------------------------------------------------------
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	// optional: de-allocate all resources once they've outlived their purpose:
	// ------------------------------------------------------------------------
	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &VBO);
	glDeleteBuffers(1, &EBO);
	//glDeleteProgram(shaderProgram);

	// glfw: terminate, clearing all previously allocated GLFW resources.
	// ------------------------------------------------------------------
	glfwTerminate();

	cudaDestroyTextureObject(cuTexture);
	cudaFreeArray(cuArray);
	return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow* window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	// make sure the viewport matches the new window dimensions; note that width and 
	// height will be significantly larger than specified on retina displays.
	glViewport(0, 0, width, height);
}