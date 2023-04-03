@echo off

if not exist "build" (
mkdir build
) else (
rmdir /s /q build
mkdir build
)

cd build
cmake ..
cmake --build . --config Release
cd ..
copy .\build\Release\c_functions.dll ..\gym_uav\env
echo copy 'c_functions.dll' to ..\gym_uav\env\