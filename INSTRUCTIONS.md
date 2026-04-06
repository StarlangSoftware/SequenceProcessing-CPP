# SequenceProcessing-CPP Build Instructions (CLion)

This guide explains how to set up and build the `SequenceProcessing-CPP` project in CLion using Conan.

## 1. Prerequisites
Ensure you have the following installed on your system:
- **C++ Compiler** (supporting C++20)
- **CMake** (v3.22 or higher)
- **Python 3** and **pip**
- **Conan 2.x**

To install Conan (if not already installed):
```bash
pip install conan
```

## 2. Configure Conan Remote
Add the Ozyegin remote to your Conan configuration (use `--force` if it already exists):
```bash
conan remote add ozyegin http://104.247.163.162:8081/artifactory/api/conan/conan-local --force
```

## 3. Install Dependencies in CLion
In the CLion **Terminal** tab (at the bottom of the IDE), run:

```bash
# Detect default profile if this is your first time using Conan 2
conan profile detect --force

# Install dependencies into the CLion build folder
conan install . --output-folder=cmake-build-debug --build=missing
```

## 4. Configure CLion CMake Settings
After running the Conan command, you need to tell CLion to use the generated toolchain.

1.  Open **Settings** (or **Preferences** on Mac) -> **Build, Execution, Deployment** -> **CMake**.
2.  In the **CMake options** field, add the following (choose the one matching your build type):
    *   For **Debug**: `-DCMAKE_TOOLCHAIN_FILE=build/Debug/generators/conan_toolchain.cmake`
    *   For **Release**: `-DCMAKE_TOOLCHAIN_FILE=build/Release/generators/conan_toolchain.cmake`

    > [!NOTE]
    > These paths are relative to the CLion build directory (e.g., `cmake-build-debug` or `cmake-build-release`).

3.  Click **Apply** and **OK**.

## 5. Reload and Build
1.  Go to **File** -> **Reload CMake Project**. (This will also automatically copy all `.txt` data files to the build directory).
2.  Once CMake finishes loading, click the **Build** button (Hammer icon) in the top toolbar.
3.  To run the tests, select the **Test** target from the run configuration dropdown and click the **Run** button (Play icon).

## 6. Verification
Run the `Test` target. If everything is set up correctly, it will execute the test cases in `Test/SequenceCorpusTest.cpp` and `Test/TransformerTest.cpp`.
