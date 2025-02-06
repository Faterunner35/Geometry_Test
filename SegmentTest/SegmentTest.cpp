// SegmentTest.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <iostream>
#include "Geomerty.h"
#include"Matrix.h"

int main()
{   
Geom_Exp::GeomCore<3,double>core;
    
auto triangles_array= Geom_Exp::readTrianglesFromFile<double>("D:/Triangles.txt");

 for (int i = 0; i < triangles_array.size()-1; i+=2)
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
