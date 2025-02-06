#pragma once
#include"Geomerty.h"
#include"Matrix.h"

template<uint32_t N,typename  T>
class Plane
{   
   public:
	Plane(const Vector<3, T>& v0, const Vector<3, T>& v1, const Vector<3, T>& v2)
	{
	   Vector<3,T>edge_1=v1-v0;
	   Vector<3,T>edge_2=v2-v1;
	   normal=Cross(edge_1,edge_2);
	   normal.Normalize();
	   distance=-DotProduct(v0,normal);
   }

	Plane() = default;
	Plane(const Plane& pl)
	{
		this->normal = pl.normal;
		this->distance = pl.distance;

	}
	Plane& operator=(const Plane& pl)
	{
		this->normal = pl.normal;
		distance = pl.distance;
		return *this;

	}

	~Plane();

	 inline T PlanePlaneIntersection(const Plane<N, T>& pl_1, const Plane<N, T>& pl_2)
	{
		Geom_Exp::Vector<N, T> cross_fo_planes = Cross(pl_1.normal, pl_2.normal);
		auto result = pl_1.distance * (Cross(pl_1.normal, cross_fo_planes)) + pl_2.distance * Cross(pl_2.normal, cross_fo_planes) / DotProduct(cross_fo_planes, cross_fo_planes);
		return result;

	}

	       
private:
   Matrix<T,3,3> world_TM;
   Vector<uint32_t N,T>normal;
   double distance;
};

