#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <iostream>
#include <functional>
#include <vector>
#include <algorithm>
#include <set>
#include <fstream>

using namespace std;

vector<const char *> validationLayers = { "VK_LAYER_LUNARG_standard_validation" };
vector<const char*> deviceExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

#ifdef NDEBUG
	const bool enableValidationLayers = false;
#else
	const bool enableValidationLayers = true;
#endif

enum class Severity { Warning, Error, CriticalError };
#define CHECK(f,s) { VkResult r = f; if(r != VK_SUCCESS){ cout << #f " failed with code " << r << endl; if(s != Severity::Warning) exit(1);} }
#define CHECK_R(f,s) { VkResult r = f; if(r != VK_SUCCESS){ cout << #f " failed with code " << r << endl; if(s != Severity::Warning) return r;} }
#define ASSERT(cond,s) { if(!(cond)){ cout << #cond " failed" << endl; if(s != Severity::Warning) exit(1);} }
#define ASSERT_R(cond,s,e) { if(!(cond)){ cout << #cond " failed" << endl; if(s != Severity::Warning) return e;} }

bool checkValidationLayerSupport()
{
	uint32_t layerCount;
	vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

	std::vector<VkLayerProperties> availableLayers(layerCount);
	vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

	for (const char* layerName : validationLayers) 
	{
		bool layerFound = false;

		for (const auto& layerProperties : availableLayers) 
		{
			if(strcmp(layerName, layerProperties.layerName) == 0) 
			{
				layerFound = true;
				break;
			}
		}

		if (!layerFound)
			return false;
	}

	return true;
}

void initWindow(GLFWwindow*& window,int Width = 1280,int Height = 720)
{
	glfwInit();

	// Tell GLFW to not use OpenGl, and also disable resize for now
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

	window = glfwCreateWindow(Width, Height, "Vulkan", nullptr, nullptr);
}

vector<const char*> getGloballyRequiredExtensions()
{
	// Just for testing, print the supported extensions
	/*uint32_t extensionCount = 0;
	vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
	vector<VkExtensionProperties> extensions(extensionCount);
	vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());
	cout << "Available extensions:" << endl;
	for (const auto& extension : extensions)
		cout << "\t" << extension.extensionName << endl;*/

	uint32_t glfwExtensionCount = 0;
	const char** glfwExtensions;
	// This pointer is managed by glfw, so need to make a copy
	glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

	ASSERT(glfwExtensions != nullptr,Severity::CriticalError);

	vector<const char*> extensions(glfwExtensionCount);
	uint32_t i = 0;
	for (; i < glfwExtensionCount; i++) 
		extensions[i] = glfwExtensions[i];

	if (enableValidationLayers)
		extensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);

	return extensions;
}

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
	VkDebugReportFlagsEXT flags,
	VkDebugReportObjectTypeEXT objType,
	uint64_t obj,
	size_t location,
	int32_t code,
	const char* layerPrefix,
	const char* msg,
	void* userData)
{
	cerr << "validation layer: " << msg << endl;

	return VK_FALSE;
}

VkResult CreateDebugReportCallbackEXT(VkInstance instance, VkDebugReportCallbackEXT* callback)
{
	VkDebugReportCallbackCreateInfoEXT createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
	createInfo.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT;
	createInfo.pfnCallback = debugCallback;

	auto func = (PFN_vkCreateDebugReportCallbackEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugReportCallbackEXT");
	if (func != nullptr)
		return func(instance, &createInfo, nullptr, callback);
	else
		return VK_ERROR_EXTENSION_NOT_PRESENT;
}

void DestroyDebugReportCallbackEXT(VkInstance instance, VkDebugReportCallbackEXT callback) 
{
	auto func = (PFN_vkDestroyDebugReportCallbackEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugReportCallbackEXT");
	if (func != nullptr)
		func(instance, callback, nullptr);
}

VkResult createInstance(VkInstance& instance)
{
	// Optional app info
	VkApplicationInfo appInfo = {};
	appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	appInfo.pApplicationName = "Hello Triangle";
	appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
	appInfo.pEngineName = "No Engine";
	appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
	appInfo.apiVersion = VK_API_VERSION_1_0;

	// Required global config
	VkInstanceCreateInfo createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	createInfo.pApplicationInfo = &appInfo;

	// Make sure there is validation layer support if it's required
	if (enableValidationLayers)
	{
		ASSERT(checkValidationLayerSupport(), Severity::CriticalError);
		createInfo.enabledLayerCount = validationLayers.size();
		createInfo.ppEnabledLayerNames = validationLayers.data();
	}

	// Get the array of required extensions
	// This are global, not device specific
	auto extensions = getGloballyRequiredExtensions();
	createInfo.enabledExtensionCount = extensions.size();
	createInfo.ppEnabledExtensionNames = extensions.data();

	return vkCreateInstance(&createInfo, nullptr, &instance);
}

VkResult createSurface(VkInstance& instance,VkSurfaceKHR& surface, GLFWwindow*& window)
{
	return glfwCreateWindowSurface(instance, window, nullptr, &surface);
}

bool checkDeviceExtensionsSupport(const VkPhysicalDevice& device, set<string> extensions)
{
	uint32_t extensionCount;
	vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

	vector<VkExtensionProperties> availableExtensions(extensionCount);
	vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

	for (const auto& extension : availableExtensions) {
		extensions.erase(extension.extensionName);
	}

	return extensions.empty();
}

struct SwapChainSupportDetails
{
	VkSurfaceCapabilitiesKHR Capabilities;
	vector<VkSurfaceFormatKHR> Formats;
	vector<VkPresentModeKHR> PresentModes;
};

// Sometimes used const, others not. Got lazy. All are pointers anyway
// Also lazy, not checking returns here :(
SwapChainSupportDetails querySwapChainSupport(const VkPhysicalDevice& device, VkSurfaceKHR& surface)
{
	SwapChainSupportDetails details;

	uint32_t formatCount;
	vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

	if (formatCount != 0)
	{
		details.Formats.resize(formatCount);
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.Formats.data());
	}
	
	uint32_t presentModeCount;
	vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

	if (presentModeCount != 0)
	{
		details.PresentModes.resize(presentModeCount);
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.PresentModes.data());
	}

	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device,surface,&details.Capabilities);

	return details;
}

VkSurfaceFormatKHR findFormat(const SwapChainSupportDetails& swapDetails,bool & swapFormatFound)
{
	// For some reason, it looks like B8G8R8A8 is supported but R8G8B8A8 is not. Who knows
	swapFormatFound = false;
	if (swapDetails.Formats.size() == 1 && swapDetails.Formats[0].format == VK_FORMAT_UNDEFINED)
	{
		swapFormatFound = true; // it supports any format
		return { VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
	}

	// Try to find a 32 bpp format with sRGB
	for (const auto& f : swapDetails.Formats)
		if (f.format == VK_FORMAT_B8G8R8A8_UNORM && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
		{
			swapFormatFound = true;
			return { VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
		}
	
	return {};
}

int rateDeviceSuitability(const VkPhysicalDevice& physicalDevice, VkSurfaceKHR& surface)
{
	VkPhysicalDeviceFeatures deviceFeatures;
	vkGetPhysicalDeviceFeatures(physicalDevice, &deviceFeatures);

	SwapChainSupportDetails swapDetails = querySwapChainSupport(physicalDevice,surface);
	bool swapFormatFound;
	findFormat(swapDetails,swapFormatFound);

	// Make sure there's tesellation, just if I want to use it later.
	// Mostly testing stuff here
	// More importantly, make sure the required extensions are supported and that the swap chain is supported
	if( !deviceFeatures.tessellationShader || 
		!deviceFeatures.geometryShader || 
		!checkDeviceExtensionsSupport(physicalDevice,set<string>(deviceExtensions.begin(),deviceExtensions.end())) ||
		!(swapFormatFound && swapDetails.PresentModes.size() > 0))
		return 0;

	int score = 1;

	VkPhysicalDeviceProperties deviceProperties;
	vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);

	// Prefer discrete GPUs
	if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
		score += 1;

	return score;
}

VkResult pickPhysicalDevice(VkInstance& instance, VkSurfaceKHR& surface, VkPhysicalDevice& physicalDevice)
{
	physicalDevice = VK_NULL_HANDLE;
	uint32_t deviceCount = 0;
	CHECK_R(vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr),Severity::CriticalError);
	
	// I couldn't find a suitable error, so using VK_RESULT_MAX_ENUM as a subsitute
	// There needs to be some "unknown" or "generic" error
	ASSERT_R(deviceCount > 0,Severity::CriticalError, VK_RESULT_MAX_ENUM);

	vector<VkPhysicalDevice> devices(deviceCount);
	CHECK_R(vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data()), Severity::CriticalError);

	// Find the best candidate physicalDevice
	// Use an ordered map to automatically sort candidates by increasing score
	// A score of 0 indicates that the physicalDevice is not suitable
	int bestScore = -1;
	for (const auto& dev : devices)
	{
		int score = rateDeviceSuitability(dev,surface);
		if(score > 0 && score > bestScore)
		{
			bestScore = score;
			physicalDevice = dev;
		}
	}
	
	// Make sure a suitable physicalDevice was found
	ASSERT_R(physicalDevice != VK_NULL_HANDLE, Severity::CriticalError, VK_RESULT_MAX_ENUM);

	return VK_SUCCESS;
}

struct DeviceQueues
{
	VkDevice LogicalDevice;
	VkQueue Compute;
	VkQueue Transfer;
	// This are actually the same queue, and I don't handle different queues in the code, but they could potentially be different
	VkQueue Graphics;
	VkQueue Present;

	uint32_t ComputeFamilyIDX, TransferFamilyIDX, GraphicsFamilyIDX, PresentFamilyIDX;
};


VkResult createDeviceQueues(VkPhysicalDevice& physicalDevice,VkSurfaceKHR& surface, DeviceQueues& queues)
{
	uint32_t queueFamilyCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);

	std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
	vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

	uint32_t queueIndexes[] = {-1,-1,-1};

	// Try to find first specialized queues, as they are usually faster
	// Except on the present queue, that could be faster if it's the same as graphics acording to the tutorial. Makes sense
	// For now, forcing the graphics queue to be also the present queue
	for(int i = 0;i < queueFamilyCount;i++)
	{
		VkBool32 presentSupported = false;
		vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, surface, &presentSupported);

		if (queueFamilies[i].queueCount > 0 && queueFamilies[i].queueFlags == VK_QUEUE_COMPUTE_BIT)
			queueIndexes[0] = i;
		else if (queueFamilies[i].queueCount > 0 && queueFamilies[i].queueFlags == VK_QUEUE_GRAPHICS_BIT && presentSupported)
			queueIndexes[1] = i;
		else if (queueFamilies[i].queueCount > 0 && queueFamilies[i].queueFlags == VK_QUEUE_TRANSFER_BIT)
			queueIndexes[2] = i;
	}

	// Check if any of the indexes are -1, and if they are search for any queue that works
	if(queueIndexes[0] == -1 || queueIndexes[1] == -1 || queueIndexes[2] == -1)
	{
		for (int i = 0; i < queueFamilyCount; i++)
		{
			VkBool32 presentSupported = false;
			vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, surface, &presentSupported);

			if (queueIndexes[0] == -1 && queueFamilies[i].queueCount > 0 && queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
				queueIndexes[0] = i;
			if (queueIndexes[1] == -1 && queueFamilies[i].queueCount > 0 && queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT && presentSupported)
				queueIndexes[1] = i;
			if (queueIndexes[2] == -1 && queueFamilies[i].queueCount > 0 && queueFamilies[i].queueFlags & VK_QUEUE_TRANSFER_BIT)
				queueIndexes[2] = i;
		}
	}

	ASSERT_R(queueIndexes[0] != -1 && queueIndexes[1] != -1 && queueIndexes[2] != -1, Severity::CriticalError, VK_RESULT_MAX_ENUM);

	// With queues indexes selected, create the logical device and get the queues handles
	queues = {};
	VkDeviceQueueCreateInfo queueCreateInfo[3] = {};
	float queuePriority = 1.0f;
	
	// Make sure there are no repeated queues
	int queueIndexesCount;
	if(queueIndexes[0] != queueIndexes[1] && queueIndexes[0] != queueIndexes[2] && queueIndexes[1] != queueIndexes[2])
		queueIndexesCount = 3;
	else
	{
		if(queueIndexes[0] == queueIndexes[1] && queueIndexes[1] == queueIndexes[2])
			queueIndexesCount = 1;
		else
			queueIndexesCount = 2;
	}

	// Create 3 infos, they may not all be used if some queues have the same index
	queueCreateInfo[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
	queueCreateInfo[0].queueFamilyIndex = queueIndexes[0];
	queueCreateInfo[0].queueCount = 1;
	queueCreateInfo[0].pQueuePriorities = &queuePriority; // <- This is only useful when there are more than one queue of the same type, but is always required
	
	// Fill the remaining ones with the unique family indexes
	for(int q = 1;q < queueIndexesCount;q++)
	{
		int index = 0;
		for(int i = 1;i < 3;i++)
		{
			bool repeated = false;
			for(int j = 0;j < q && !repeated;j++)
				if(queueIndexes[i] == queueCreateInfo[j].queueFamilyIndex)
					repeated = true;
			if(repeated) continue;
			index = queueIndexes[i];
		}

		queueCreateInfo[q].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queueCreateInfo[q].queueFamilyIndex = index;
		queueCreateInfo[q].queueCount = 1;
		queueCreateInfo[q].pQueuePriorities = &queuePriority;
	}

	// Adding some required features just for fun
	VkPhysicalDeviceFeatures deviceFeatures = {};
	deviceFeatures.tessellationShader = true;
	VkDeviceCreateInfo createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	createInfo.pQueueCreateInfos = queueCreateInfo;
	createInfo.queueCreateInfoCount = queueIndexesCount;
	createInfo.pEnabledFeatures = &deviceFeatures;
	createInfo.enabledExtensionCount = deviceExtensions.size();
	createInfo.ppEnabledExtensionNames = deviceExtensions.data();

	if (enableValidationLayers)
	{
		createInfo.enabledLayerCount = (uint32_t) validationLayers.size();
		createInfo.ppEnabledLayerNames = validationLayers.data();
	}

	CHECK_R(vkCreateDevice(physicalDevice, &createInfo, nullptr, &queues.LogicalDevice), Severity::CriticalError);
	vkGetDeviceQueue(queues.LogicalDevice, queueIndexes[0], 0, &queues.Compute);
	vkGetDeviceQueue(queues.LogicalDevice, queueIndexes[1], 0, &queues.Graphics);
	vkGetDeviceQueue(queues.LogicalDevice, queueIndexes[2], 0, &queues.Transfer);
	queues.ComputeFamilyIDX = queueIndexes[0];
	queues.GraphicsFamilyIDX = queueIndexes[1];
	queues.TransferFamilyIDX = queueIndexes[2];

	// Like I said before, using the same handle for graphics and present, but they could potentially be different
	queues.Present = queues.Graphics;
	queues.PresentFamilyIDX = queues.GraphicsFamilyIDX;

	return VK_SUCCESS;
}

VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR> availablePresentModes) 
{
	// FIFO = Standard VSync (with wait)
	// MAILBOX = Triple Buffering (less wait, no tearing)
	// IMMEDIATE = Single buffer (no wait, tearing)
	VkPresentModeKHR bestMode = VK_PRESENT_MODE_FIFO_KHR;

	// From https://vulkan-tutorial.com/Drawing_a_triangle/Presentation/Swap_chain
	//Unfortunately some drivers currently don't properly support VK_PRESENT_MODE_FIFO_KHR, 
	//	so we should prefer VK_PRESENT_MODE_IMMEDIATE_KHR if VK_PRESENT_MODE_MAILBOX_KHR is not available:
	for (const auto& availablePresentMode : availablePresentModes) 
	{
		if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
			return availablePresentMode;
		else if (availablePresentMode == VK_PRESENT_MODE_IMMEDIATE_KHR)
			bestMode = availablePresentMode;
	}

	return bestMode;
}

VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities, uint32_t sizeX, uint32_t sizeY) 
{
	// If the capabilities return maxint as the current extent, then it needs to be provided
	// Otherwise it's taken from the window size
	if (capabilities.currentExtent.width != numeric_limits<uint32_t>::max())
		return capabilities.currentExtent;
	else 
	{
		VkExtent2D actualExtent = { sizeX, sizeY };

		actualExtent.width = max(capabilities.minImageExtent.width, min(capabilities.maxImageExtent.width, actualExtent.width));
		actualExtent.height = max(capabilities.minImageExtent.height, min(capabilities.maxImageExtent.height, actualExtent.height));

		return actualExtent;
	}
}

VkResult createSwapChain(VkPhysicalDevice& physicalDevice,VkSurfaceKHR& surface,DeviceQueues& queues,uint32_t sizeX, uint32_t sizeY, VkSwapchainKHR& swapChain, VkExtent2D& outExtent, vector<VkImage>& swapChainImages)
{
	SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice, surface);
	bool swapFormatFound;
	VkSurfaceFormatKHR surfaceFormat = findFormat(swapChainSupport, swapFormatFound); // we already know the format can be found
	VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.PresentModes);
	outExtent = chooseSwapExtent(swapChainSupport.Capabilities, sizeX,sizeY);

	// From : https://vulkan-tutorial.com/Drawing_a_triangle/Presentation/Swap_chain
	// A value of 0 for maxImageCount means that there is no limit besides memory requirements, 
	//		which is why we need to check for that.
	uint32_t imageCount = swapChainSupport.Capabilities.minImageCount + 1;
	if (swapChainSupport.Capabilities.maxImageCount > 0 && imageCount > swapChainSupport.Capabilities.maxImageCount)
		imageCount = swapChainSupport.Capabilities.maxImageCount;

	VkSwapchainCreateInfoKHR createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
	createInfo.surface = surface;
	createInfo.minImageCount = imageCount;
	createInfo.imageFormat = surfaceFormat.format;
	createInfo.imageColorSpace = surfaceFormat.colorSpace;
	createInfo.imageExtent = outExtent;
	createInfo.imageArrayLayers = 1; // for stereo 3D
	createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT; // equivalent to DX RTV usage

	// Not really necessary, as I'm forcing this too to be the same
	//	but it doesn't hurt for the future
	uint32_t queueFamilyIndices[2] = { queues.GraphicsFamilyIDX, queues.PresentFamilyIDX };
	if (queues.GraphicsFamilyIDX != queues.PresentFamilyIDX) 
	{
		createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
		createInfo.queueFamilyIndexCount = 2;
		createInfo.pQueueFamilyIndices = queueFamilyIndices;
	}
	else 
		createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;

	// Swapchain transforms, like screen rotation
	createInfo.preTransform = swapChainSupport.Capabilities.currentTransform;
	// Blending with other windows
	createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	createInfo.presentMode = presentMode;
	// Ignore pixels that are obscured by other windows
	createInfo.clipped = VK_TRUE;
	// Used when resizing and stuff. Ignored for now
	createInfo.oldSwapchain = VK_NULL_HANDLE;

	CHECK_R(vkCreateSwapchainKHR(queues.LogicalDevice, &createInfo, nullptr, &swapChain), Severity::CriticalError);

	vkGetSwapchainImagesKHR(queues.LogicalDevice, swapChain, &imageCount, nullptr);
	swapChainImages.resize(imageCount);
	return vkGetSwapchainImagesKHR(queues.LogicalDevice, swapChain, &imageCount, swapChainImages.data());
}

VkResult createImageViews(vector<VkImage>& swapChainImages, DeviceQueues& queues, vector<VkImageView>& swapChainImageViews)
{
	swapChainImageViews.resize(swapChainImages.size());

	for (size_t i = 0; i < swapChainImages.size(); i++) 
	{
		VkImageViewCreateInfo createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		createInfo.image = swapChainImages[i];
		createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		createInfo.format = VK_FORMAT_B8G8R8A8_UNORM; // check findFormat for reference

		// Allows to swizzle or collapse the channels
		createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
		createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
		createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
		createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

		createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; // <- What stuff to clear
		createInfo.subresourceRange.baseMipLevel = 0;
		createInfo.subresourceRange.levelCount = 1;
		createInfo.subresourceRange.baseArrayLayer = 0;
		createInfo.subresourceRange.layerCount = 1;

		CHECK_R(vkCreateImageView(queues.LogicalDevice, &createInfo, nullptr, &swapChainImageViews[i]), Severity::CriticalError);
	}

	return VK_SUCCESS;
}

vector<char> readFile(const string& filename) 
{
	ifstream file(filename, ios::ate | ios::binary);

	if (!file.is_open())
		return vector<char>();

	size_t size = file.tellg();
	vector<char> data(size);

	file.seekg(0);
	file.read(data.data(),size);

	file.close();
	return data;
}

VkShaderModule createShaderModule(DeviceQueues& queues, const vector<char>& code)
{
	VkShaderModuleCreateInfo createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	createInfo.codeSize = code.size();
	createInfo.pCode = (uint32_t*)code.data();

	VkShaderModule shaderModule;
	ASSERT_R(vkCreateShaderModule(queues.LogicalDevice, &createInfo, nullptr, &shaderModule) != VK_SUCCESS, Severity::CriticalError, nullptr);

	return shaderModule;
}

struct Pipeline
{
	VkShaderModule VertexShader;
	VkShaderModule FragmentShader;
	VkPipelineLayout Layout;

	VkPipelineShaderStageCreateInfo VertexInfo,FragmentInfo;
};

VkResult createGraphicsPipeline(DeviceQueues& queues,VkExtent2D& extent, Pipeline& pipeline)
{
	auto vertShaderCode = readFile("vert.spv");
	auto fragShaderCode = readFile("frag.spv");

	ASSERT_R(vertShaderCode.size() > 0 && fragShaderCode.size() > 0, Severity::CriticalError, VK_RESULT_MAX_ENUM);
	pipeline.VertexShader = createShaderModule(queues,vertShaderCode);
	pipeline.FragmentShader = createShaderModule(queues, vertShaderCode);

	ASSERT_R(pipeline.VertexShader != nullptr && pipeline.FragmentShader != nullptr, Severity::CriticalError, VK_RESULT_MAX_ENUM);
	
	pipeline.VertexInfo = {};
	pipeline.VertexInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	pipeline.VertexInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
	pipeline.VertexInfo.module = pipeline.VertexShader;
	pipeline.VertexInfo.pName = "main";
	// Not using pSpecializationInfo for now, that allows to set shader constants at compile time, so leaving it at null

	pipeline.FragmentInfo = {};
	pipeline.FragmentInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	pipeline.FragmentInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	pipeline.FragmentInfo.module = pipeline.FragmentShader;
	pipeline.FragmentInfo.pName = "main";

	// Create fixed pipeline stuff
	// Not using any vertex buffer for now
	VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
	vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertexInputInfo.vertexBindingDescriptionCount = 0;
	vertexInputInfo.pVertexBindingDescriptions = nullptr; // Optional
	vertexInputInfo.vertexAttributeDescriptionCount = 0;
	vertexInputInfo.pVertexAttributeDescriptions = nullptr; // Optional

	VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
	inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	inputAssembly.primitiveRestartEnable = VK_FALSE; // for the strips modes, not using that

	VkViewport viewport = {};
	viewport.x = 0.0f;
	viewport.y = 0.0f;
	viewport.width = (float) extent.width;
	viewport.height = (float) extent.height;
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;
	VkRect2D scissor = {};
	scissor.offset = { 0, 0 };
	scissor.extent = extent;
	VkPipelineViewportStateCreateInfo viewportState = {};
	viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportState.viewportCount = 1;
	viewportState.pViewports = &viewport;
	viewportState.scissorCount = 1;
	viewportState.pScissors = &scissor;

	VkPipelineRasterizationStateCreateInfo rasterizer = {};
	rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterizer.depthClampEnable = VK_FALSE;
	rasterizer.rasterizerDiscardEnable = VK_FALSE;
	rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
	rasterizer.lineWidth = 1.0f; // line width in fragments
	rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
	rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
	rasterizer.depthBiasEnable = VK_FALSE;
	rasterizer.depthBiasConstantFactor = 0.0f; // Optional
	rasterizer.depthBiasClamp = 0.0f; // Optional
	rasterizer.depthBiasSlopeFactor = 0.0f; // Optional
	
	// Not using MSAA for now
	VkPipelineMultisampleStateCreateInfo multisampling = {};
	multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisampling.sampleShadingEnable = VK_FALSE;
	multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
	multisampling.minSampleShading = 1.0f; // Optional
	multisampling.pSampleMask = nullptr; // Optional
	multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
	multisampling.alphaToOneEnable = VK_FALSE; // Optional

	// Depth and stencil buffer creation would go here

	// Set blend (to disabled)
	//		Global
	VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
	colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	colorBlendAttachment.blendEnable = VK_FALSE;
	colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
	colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
	colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD; // Optional
	colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
	colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
	colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD; // Optional
	//		Per backbuffer
	VkPipelineColorBlendStateCreateInfo colorBlending = {};
	colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	colorBlending.logicOpEnable = VK_FALSE;
	colorBlending.logicOp = VK_LOGIC_OP_COPY; // Optional
	colorBlending.attachmentCount = 1;
	colorBlending.pAttachments = &colorBlendAttachment;
	colorBlending.blendConstants[0] = 0.0f; // Optional
	colorBlending.blendConstants[1] = 0.0f; // Optional
	colorBlending.blendConstants[2] = 0.0f; // Optional
	colorBlending.blendConstants[3] = 0.0f; // Optional

	VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
	pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutInfo.setLayoutCount = 0; // Optional
	pipelineLayoutInfo.pSetLayouts = nullptr; // Optional
	pipelineLayoutInfo.pushConstantRangeCount = 0; // Optional
	pipelineLayoutInfo.pPushConstantRanges = 0; // Optional

	CHECK_R(vkCreatePipelineLayout(queues.LogicalDevice, &pipelineLayoutInfo, nullptr, &pipeline.Layout),Severity::CriticalError);

	return VK_SUCCESS;
}

int main() 
{
	int Width = 1280;
	int Height = 720;

	// Create window
	GLFWwindow* window;
	initWindow(window, Width, Height);

	// Create vulkan instance
	VkInstance instance;
	CHECK(createInstance(instance),Severity::CriticalError);
	
	// Create surface
	VkSurfaceKHR surface;
	CHECK(createSurface(instance,surface,window), Severity::CriticalError);

	// Create validation layer callback
	VkDebugReportCallbackEXT validationLayerCallback;
	if(enableValidationLayers)
		CHECK(CreateDebugReportCallbackEXT(instance,&validationLayerCallback),Severity::CriticalError);

	// Select physical device
	VkPhysicalDevice physicalDevice;
	CHECK(pickPhysicalDevice(instance,surface,physicalDevice), Severity::CriticalError);

	// Create queues
	DeviceQueues queues;
	CHECK(createDeviceQueues(physicalDevice,surface,queues), Severity::CriticalError);

	// Create swapchain
	VkSwapchainKHR swapchain;
	vector<VkImage> swapChainImages;
	VkExtent2D  extent;
	CHECK(createSwapChain(physicalDevice, surface, queues, Width, Height, swapchain,extent, swapChainImages), Severity::CriticalError);

	// Create swapchain image views
	vector<VkImageView> swapChainImageViews;
	CHECK(createImageViews(swapChainImages,queues,swapChainImageViews), Severity::CriticalError);

	// Create pipeline
	Pipeline pipeline;
	CHECK(createGraphicsPipeline(queues,extent,pipeline), Severity::CriticalError);
	
	// Main loop
	while (!glfwWindowShouldClose(window)) 
	{
		glfwPollEvents();
	}

	// Cleanup
	vkDestroyPipelineLayout(queues.LogicalDevice, pipeline.Layout, nullptr);
	vkDestroyShaderModule(queues.LogicalDevice, pipeline.FragmentShader, nullptr);
	vkDestroyShaderModule(queues.LogicalDevice, pipeline.VertexShader, nullptr);
	for (auto imageView : swapChainImageViews)
		vkDestroyImageView(queues.LogicalDevice, imageView, nullptr);
	vkDestroySwapchainKHR(queues.LogicalDevice, swapchain, nullptr);
	vkDestroySurfaceKHR(instance, surface, nullptr);
	vkDestroyDevice(queues.LogicalDevice, nullptr);
	DestroyDebugReportCallbackEXT(instance,validationLayerCallback);
	vkDestroyInstance(instance, nullptr);
	glfwDestroyWindow(window);
	glfwTerminate();
	return EXIT_SUCCESS;
}