// SegmentTest.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <iostream>
#include <string>
#include "Geomerty.h"
#include"Matrix.h"

int main(int argc, char* argv[])
{   
Geom_Exp::GeomCore<3,double>core;

// Input file: first command-line argument; the original hard-coded path is kept as the default.
const std::string input_path = (argc > 1) ? argv[1] : "D:/Triangles.txt";
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

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
