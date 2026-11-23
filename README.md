# Geometry Test: triangle–triangle intersection

[![build](https://github.com/Faterunner35/Geometry_Test/actions/workflows/build.yml/badge.svg)](https://github.com/Faterunner35/Geometry_Test/actions/workflows/build.yml)

A small header-based C++ geometry experiment: a templated `Vector<N,T>`, segment and triangle types, closest points between two segments, and a test for whether two triangles in 3D intersect. The console program `SegmentTest` reads pairs of triangles from a text file and reports for each pair whether they intersect.

This is early experimental code. Parts of the headers are unfinished (see *Status* below).

## Files

| File | Contents |
|---|---|
| `src/Geometry.h` | Everything that is actually used, in namespace `Geom_Exp`: `Vector<N,T>` with arithmetic, `DotProduct`, `Cross`, `UnitCross`, `Normalize`, `Orthonormalize` and lift/project helpers; `Segment<N,T>`; `Triangle<T>`; `readTrianglesFromFile<T>()`; `GeomCore<N,T>` with segment–segment closest points (`operator()`), `Intersects()` (where triangle V's edges cross the plane of triangle U) and `TriangleIntersection()` |
| `src/main.cpp` | `main()`: reads triangles, runs `TriangleIntersection` on each consecutive pair, prints the result |
| `src/Geometry.cpp` | Includes `Geometry.h` only (compile check of the header) |
| `src/Matrix.h` | Draft `Matrix<T,Column,Rows>` template; included but not used |
| `src/Plane.h` | Draft `Plane` class; not included anywhere and does not compile as written |
| `data/Triangles.txt` | Sample input with five triangle pairs |
| `CMakeLists.txt`, `CMakePresets.json` | CMake build (Windows/MSVC, Linux/GCC) |

## How the intersection test works

`TriangleIntersection(U, V)` handles triangles that are not coplanar:

1. `Intersects(U, V, S0)` computes the signed distances of V's vertices to the plane of U. If V has vertices strictly on both sides, the part of V lying in that plane is a segment `S0`, found by linear interpolation along V's edges. Otherwise the function returns false, so a triangle that only touches the plane at a vertex or along an edge does not count as intersecting.
2. `Intersects(V, U, S1)` does the same with the roles swapped.
3. Both segments lie on the line where the two planes meet. Projecting their endpoints onto the line direction `cross(nU, nV)` gives two intervals, and the triangles intersect if the intervals overlap.

There is no separate handling of coplanar triangles: when all signed distances are zero, the result is `false`.

## Input format

One triangle per line, nine numbers: `x0 y0 z0  x1 y1 z1  x2 y2 z2`. Blank lines and lines starting with `#` are skipped. Lines are taken in pairs: lines 1–2 are the first test, lines 3–4 the second, and so on.

## Build

Requires a C++20 compiler (MSVC 2022, GCC ≥ 11, Clang ≥ 14). The CMake build also needs CMake ≥ 3.20.

### Visual Studio 2022

*File → Open → Folder…* on the repository root: Visual Studio reads `CMakeLists.txt` and `CMakePresets.json` (presets `x64 Debug` / `x64 Release`). To run on the sample file, set the working directory to the repository root or pass `data\Triangles.txt` as the argument. The old `.sln`/`.vcxproj` were removed: CMake generates them when needed (`cmake -G "Visual Studio 17 2022" -B build`).

### CMake (any platform)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

Or with presets: `cmake --preset linux-release && cmake --build --preset linux-release && ctest --preset linux-release` (on Windows: `x64-release`).

## Run

```bash
./build/SegmentTest data/Triangles.txt
#   Windows: build\Release\SegmentTest.exe data\Triangles.txt
```

Without an argument the program reads `data/Triangles.txt` relative to the working directory, so running it from the repository root uses the sample file.

Output for the sample file:

```
Test number1Passed
Test number3Failed
Test number5Failed
Test number7Passed
Test number9Failed
```

`Passed` means the pair intersects and `Failed` means it does not. The number is the 1-based index of the pair's first triangle (1, 3, 5, …), not counting comment lines. The CTest test `triangle_pairs` checks exactly this output.

## Credits

Parts of `src/Geometry.h` are adapted from David Eberly's [Geometric Tools](https://www.geometrictools.com) (GTE): the segment–segment closest-point query (its region-based case analysis) and the structure of the triangle–triangle intersection test follow the GTE implementations. Geometric Tools is distributed under the [Boost Software License 1.0](https://www.boost.org/LICENSE_1_0.txt), which asks that its copyright notice be kept with derived code:

> David Eberly, Geometric Tools, Redmond WA 98052 — Copyright (c) 1998-2023. Distributed under the Boost Software License, Version 1.0.

## Status

Known limitations of the current code, left as they are:

- `GeomCore::ProjectPointOnSegment` and `GeomCore::DistanceSegment` treat `segment.p[1]` as a direction vector, while `Segment` stores two endpoints. `GeomCore::operator()` is the correct segment–segment distance.
- `Lenght(v, robust = true)` returns inside the first loop iteration and scales by the wrong component. `GetOrthogonal` compares `v[0]` on every iteration instead of `v[i]`.
- The `Vector(std::initializer_list<T>&)` constructor takes a non-const reference, so it cannot be called with a braced list.
- `Matrix.h` and `Plane.h` are drafts (see the table above).
- `TriangleIntersection` stores the interval end points in `float`.
