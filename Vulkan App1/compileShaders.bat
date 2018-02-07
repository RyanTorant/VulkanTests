@echo off
echo Compiling shaders
C:/VulkanSDK/1.0.65.0/Bin/glslangValidator.exe -V vertex.vert
C:/VulkanSDK/1.0.65.0/Bin/glslangValidator.exe -V pixel.frag
echo Compile finished
echo ~~~~~~~~~~~~~~~~~~~~~~