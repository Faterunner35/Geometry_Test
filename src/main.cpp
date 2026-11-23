// SegmentTest: reads pairs of triangles and reports whether each pair intersects.
// Usage: SegmentTest [triangles.txt]   (default: data/Triangles.txt, relative to the working directory)
#include <iostream>
#include <string>
#include "Geometry.h"
#include"Matrix.h"

int main(int argc, char* argv[])
{   
Geom_Exp::GeomCore<3,double>core;

// Input file: first command-line argument, or the sample file when run from the repository root.
const std::string input_path = (argc > 1) ? argv[1] : "data/Triangles.txt";
auto triangles_array= Geom_Exp::readTrianglesFromFile<double>(input_path);

 for (size_t i = 0; i + 1 < triangles_array.size(); i+=2)
 {
	 bool result=core.TriangleIntersection(triangles_array[i].GetArray(), triangles_array[i+1].GetArray());
	 if(result)
		 std::cout << "Test number" << i + 1 << "Passed"<< std::endl;
		 else
		 std::cout << "Test number" << i + 1 << "Failed" << std::endl;
	
  }
     
     
    return 0;
}
