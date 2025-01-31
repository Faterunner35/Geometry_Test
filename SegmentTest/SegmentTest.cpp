// SegmentTest.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "Geomerty.h"
#include"Matrix.h"

int main()
{   
   Geom_Exp::Vector<3,double> v1({0.025,1.3,-4.5667});
   Geom_Exp::Vector<3, double> v2({ 2.025,-3.5,4.2334 });
   Geom_Exp::Vector<3, double> v3({ 34.025,-6.67,45.5667 });
   Geom_Exp::Vector<3, double> v4({-10.025,7.5,42.2334 });
   Geom_Exp::Segment<3,double> sg_1{v1,v2};
   Geom_Exp::Segment<3, double> sg_2{ v3,v4 };

   Geom_Exp::GeomCore<3,double>core;
    
   auto dist_1=core.ProjectPointOnSegment(sg_1,v3);
   auto dist_2=core.DistanceSegment(sg_1,sg_2);

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
