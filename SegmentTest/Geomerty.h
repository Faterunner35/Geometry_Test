#pragma once
#include<array>
#include<iostream>
#include<tuple>
#include<algorithm>
#include<vector>
#include<fstream>
#include<string>
#include<sstream>


namespace Geom_Exp
{ 
template<uint32_t N,typename T >
class Vector
{
  public:
      
	Vector()=default;

	Vector(std::array<T,N>const& values)
		:
		Vector_Array(values)
	{
	}

	Vector(std::initializer_list<T>&data)
	{    
		   int32_t const numValues = static_cast<int32_t>(data.size());
		   if(N==data.size)
		   {
			   std::copy(data.begin(), data.end(), Vector_Array.data());
			}
		    
			 else if(N>data.size())
			{
			    std::copy(data.begin(),data.end(),Vector_Array.data());
				std::fill_n(Vector.begin()+numValues,Vector.end(),float(0));

			}

			else
			{
			 std::copy(data.begin(),data.end()+numValues,Vector_Array.begin());

			}
	}

	inline int32_t GetSize() const
	{
		return N;
	}

	inline T const& operator[](int32_t i) const
	{
		return Vector_Array[i];
	}

	inline T& operator[](int32_t i)
	{
		return Vector_Array[i];
	}
	inline void SetToVector(uint8_t pos, T data)
	{
		Vector_Array[pos]=data;
	}

	inline auto GetVectorArray()
	{
		return Vector_Array;
	}
			
	// Special vectors.

	// All components are 0.
	void MakeZero()
	{
		std::fill(Vector_Array.begin(), Vector_Array.end(), (float)0);
	}

	// All components are 1.
	void MakeOnes()
	{
		std::fill(Vector_Array.begin(), Vector_Array.end(), (float)1);
	}

	// Component d is 1, all others are zero.
	void MakeUnit(int32_t d)
	{
		std::fill(Vector_Array.begin(), Vector_Array.end(), (float)0);
		if (0 <= d && d < N)
		{
			Vector[d] = (float)1;
		}
	}

	static Vector Zero()
	{
		Vector<N,T> v;
		v.MakeZero();
		return v;
	}

	static Vector Ones()
	{
		Vector<N, T> v;
		v.MakeOnes();
		return v;
	}

	static Vector Unit(int32_t d)
	{
		Vector<N, T> v;
		v.MakeUnit(d);
		return v;
	}

	 bool operator==(Vector const& vec) const
	{
		return this->Vector_Array == vec.Vector_Array;
	}

	 bool operator!=(Vector const& vec) const
	{
		return this->Vector_Array != vec.Vector_Array;
	}

	 bool operator< (Vector const& vec) const
	{
		return this->Vector_Array < vec.Vector_Array;
	}

	 bool operator<=(Vector const& vec) const
	{
		return this->Vector_Array <= vec.Vector_Array;
	}

	 bool operator> (Vector const& vec) const
	{
		return this->Vector_Array > vec.Vector_Array;
	}

	 bool operator>=(Vector const& vec) const
	{
		return this->Vector_Array >= vec.Vector_Array;
	}
	
	
 private:
 mutable std::array<T,N>Vector_Array;

};

template <int32_t N, typename T>
Vector<N, T> operator+(Vector<N, T> const& v)
{
	return v;
}

template <int32_t N, typename Real>
Vector<N, Real> operator-(Vector<N, Real> const& v)
{
	Vector<N, Real> result;
	for (int32_t i = 0; i < N; ++i)
	{
		result[i] = -v[i];
	}
	return result;
}

template <int32_t N, typename T>
Vector<N, T> operator+(Vector<N, T> const& v0, Vector<N, T> const& v1)
{
	Vector<N, T> result = v0;
	return result += v1;
}

template <int32_t N, typename T>
Vector<N, T> operator-(Vector<N, T> const& v0, Vector<N, T> const& v1)
{
	Vector<N,T> result = v0;
	return result -= v1;
}

template <int32_t N, typename T>
Vector<N,T> operator*(Vector<N, T> const& v, T scalar)
{
	Vector<N, T> result = v;
	return result *= scalar;
}

template <int32_t N, typename T>
Vector<N, T> operator*(T scalar, Vector<N, T> const& v)
{
	Vector<N, T> result = v;
	return result *= scalar;
}

template <int32_t N, typename T>
Vector<N, T> operator/(Vector<N, T> const& v, T scalar)
{
	Vector<N, T> result = v;
	return result /= scalar;
}

template <int32_t N, typename Real>
Vector<N, Real>& operator+=(Vector<N, Real>& v0, Vector<N, Real> const& v1)
{
	for (int32_t i = 0; i < N; ++i)
	{
		v0[i] += v1[i];
	}
	return v0;
}

template <int32_t N, typename Real>
Vector<N, Real>& operator-=(Vector<N, Real>& v0, Vector<N, Real> const& v1)
{
	for (int32_t i = 0; i < N; ++i)
	{
		v0[i] -= v1[i];
	}
	return v0;
}

template <int32_t N, typename Real>
Vector<N, Real>& operator*=(Vector<N, Real>& v, Real scalar)
{
	for (int32_t i = 0; i < N; ++i)
	{
		v[i] *= scalar;
	}
	return v;
}


template <int32_t N, typename Real>
Vector<N, Real>& operator/=(Vector<N, Real>& v, Real scalar)
{
	if (scalar != (Real)0)
	{
		Real invScalar = (Real)1 / scalar;
		for (int32_t i = 0; i < N; ++i)
		{
			v[i] *= invScalar;
		}
	}
	else
	{
		for (int32_t i = 0; i < N; ++i)
		{
			v[i] = (Real)0;
		}
	}
	return v;
}

// Componentwise algebraic operations.
template <int32_t N, typename Real>
Vector<N, Real> operator*(Vector<N, Real> const& v0, Vector<N, Real> const& v1)
{
	Vector<N, Real> result = v0;
	return result *= v1;
}

template <int32_t N, typename Real>
Vector<N, Real> operator/(Vector<N, Real> const& v0, Vector<N, Real> const& v1)
{
	Vector<N, Real> result = v0;
	return result /= v1;
}

template <int32_t N, typename Real>
Vector<N, Real>& operator*=(Vector<N, Real>& v0, Vector<N, Real> const& v1)
{
	for (int32_t i = 0; i < N; ++i)
	{
		v0[i] *= v1[i];
	}
	return v0;
}

template<uint32_t N,typename T>
T DotProduct(const Vector<N,T>& v0, const Vector<N,T>& v1)
{
    
	T dot=v0[0]*v1[0];
	for (int i = 0; i < N; i++)
	{
		dot+=v0[i]*v1[i];
	}
	return dot;
}

template<uint32_t N, typename T>
T Lenght(const Vector<N, T>& v,bool robust=false)
{
	if (robust)
	{
		T abs_max=std::fabs(v[0]);

		for (int i = 1; i < N; i++)
		{
		   T abs_curr=std::fabs(v[i]);

		   if (abs_curr > abs_max)
		   {
			   abs_max=abs_curr;
		   }
		   
		  T result;

		  if (abs_max > T(0))
		  {
			  Vector<N,T>lenght=v/abs_max;
			  result=abs_curr*std::sqrt(DotProduct(lenght,lenght));
		  }

		  else
		  {
			  result=0;
		  }
		  return result;

		}
	}
	else
	{
		return std::sqrt(DotProduct(v,v));
	}

}


template <int32_t N, typename Real>
Real Normalize(Vector<N, Real>& v, bool robust = false)
{
	if (robust)
	{
		Real maxAbsComp = std::fabs(v[0]);
		for (int32_t i = 1; i < N; ++i)
		{
			Real absComp = std::fabs(v[i]);
			if (absComp > maxAbsComp)
			{
				maxAbsComp = absComp;
			}
		}

		Real length;
		if (maxAbsComp > (Real)0)
		{
			v /= maxAbsComp;
			length = std::sqrt(DotProduct(v, v));
			v /= length;
			length *= maxAbsComp;
		}
		else
		{
			length = (Real)0;
			for (int32_t i = 0; i < N; ++i)
			{
				v[i] = (Real)0;
			}
		}
		return length;
	}
	else
	{
		Real length = std::sqrt(DotProduct(v, v));
		if (length > (Real)0)
		{
			v /= length;
		}
		else
		{
			for (int32_t i = 0; i < N; ++i)
			{
				v[i] = (Real)0;
			}
		}
		return length;
	}
}

template <typename T,uint32_t N>
T Orthonormalize(int32_t numInputs, Vector<N, T>* v, bool robust = false)
{
	if (v && 1 <= numInputs && numInputs <= N)
	{
		T minlenght=Normalize(v,robust);

		for (int i = 0; i < numInputs; i++)
		{
			for (int j = 0; j < i; j++)
			{
				T dot =DotProduct(v[i],v[j]);
				v[i]-=v[j]*dot;
			}

			T length = Normalize(v[i], robust);
			if (length < minlenght)
			{
				minlenght = length;
			}
		}
		return minlenght;
	}
	return T(0);

}


template <int32_t N, typename T>
Vector< N,T> GetOrthogonal(Vector<N, T> const& v, bool unitLength)
{
  
  T max=std::fabs(v[0]);
  uint32_t maxnum=0;
  for(int i=0;i<N;i++)
  {
       T maxcurr=std::fabs(v[0]);
	   if(maxcurr>max)
	   {
	       max=maxcurr;
           maxnum=i;  
        }

   }

   Vector<N,T>temp_vec;
   temp_vec.MakeZero();
   int32_t inext =maxnum + 1;
   if (inext == N)
   {
	   inext = 0;
   }
   temp_vec[maxnum]=v[inext];
   temp_vec[inext]=-v[maxnum];
     
   if (unitLength)
   {
	   T sqrDistance = temp_vec[maxnum] * temp_vec[maxnum] + temp_vec[inext] * temp_vec[inext];
	   T invLength = ((T)1) / std::sqrt(sqrDistance);
	   temp_vec[maxnum] *= invLength;
	   temp_vec[inext] *= invLength;
   }
   return temp_vec;

   }

template <int32_t N, typename T>
Vector<N, T> Cross(Vector<N, T> const& v0, Vector<N, T> const& v1)
{
	static_assert(N == 3 || N == 4, "Dimension must be 3 or 4.");

	Vector<N, T> result;
	result.MakeZero();
	result[0] = v0[1] * v1[2] - v0[2] * v1[1];
	result[1] = v0[2] * v1[0] - v0[0] * v1[2];
	result[2] = v0[0] * v1[1] - v0[1] * v1[0];
	return result;
}

template <int32_t N, typename T>
Vector<N,T> UnitCross(Vector<N, T> const& v0, Vector<N,T> const& v1, bool robust = false)
{
	static_assert(N == 3 || N == 4, "Dimension must be 3 or 4.");
	Vector<N, T> unitCross = Cross(v0, v1);
	Normalize(unitCross, robust);
	return unitCross;
}

template <int32_t N, typename T>
T DotCross(Vector<N, T> const& v0, Vector<N, T> const& v1,
	Vector<N, T> const& v2)
{
	static_assert(N == 3 || N == 4, "Dimension must be 3 or 4.");
	return DotProduct(v0, Cross(v1, v2));
}

template <int32_t N, typename Real>
bool ComputeExtremes(int32_t numVectors, Vector<N, Real> const* v,
	Vector<N, Real>& vmin, Vector<N, Real>& vmax)
{
	if (v && numVectors > 0)
	{
		vmin = v[0];
		vmax = vmin;
		for (int32_t j = 1; j < numVectors; ++j)
		{
			Vector<N, Real> const& vec = v[j];
			for (int32_t i = 0; i < N; ++i)
			{
				if (vec[i] < vmin[i])
				{
					vmin[i] = vec[i];
				}
				else if (vec[i] > vmax[i])
				{
					vmax[i] = vec[i];
				}
			}
		}
		return true;
	}

	return false;
}

// Lift n-tuple v to homogeneous (n+1)-tuple (v,last).
template <int32_t N, typename Real>
Vector<N + 1, Real> HLift(Vector<N, Real> const& v, Real last)
{
	Vector<N + 1, Real> result;
	for (int32_t i = 0; i < N; ++i)
	{
		result[i] = v[i];
	}
	result[N] = last;
	return result;
}

// Project homogeneous n-tuple v = (u,v[n-1]) to (n-1)-tuple u.
template <int32_t N, typename Real>
Vector<N - 1, Real> HProject(Vector<N, Real> const& v)
{
	static_assert(N >= 2, "Invalid dimension.");
	Vector<N - 1, Real> result;
	for (int32_t i = 0; i < N - 1; ++i)
	{
		result[i] = v[i];
	}
	return result;
}

// Lift n-tuple v = (w0,w1) to (n+1)-tuple u = (w0,u[inject],w1).  By
// inference, w0 is a (inject)-tuple [nonexistent when inject=0] and w1 is
// a (n-inject)-tuple [nonexistent when inject=n].
template <int32_t N, typename Real>
Vector<N + 1, Real> Lift(Vector<N, Real> const& v, int32_t inject, Real value)
{
	Vector<N + 1, Real> result;
	int32_t i;
	for (i = 0; i < inject; ++i)
	{
		result[i] = v[i];
	}
	result[i] = value;
	int32_t j = i;
	for (++j; i < N; ++i, ++j)
	{
		result[j] = v[i];
	}
	return result;
}

// Project n-tuple v = (w0,v[reject],w1) to (n-1)-tuple u = (w0,w1).  By
// inference, w0 is a (reject)-tuple [nonexistent when reject=0] and w1 is
// a (n-1-reject)-tuple [nonexistent when reject=n-1].
template <int32_t N, typename Real>
Vector<N - 1, Real> Project(Vector<N, Real> const& v, int32_t reject)
{
	static_assert(N >= 2, "Invalid dimension.");
	Vector<N - 1, Real> result;
	for (int32_t i = 0, j = 0; i < N - 1; ++i, ++j)
	{
		if (j == reject)
		{
			++j;
		}
		result[i] = v[j];
	}
	return result;
}


   template <int32_t N, typename T>
   	class Segment
	{
	public:
	
		Segment()
		{
			p[1].MakeUnit(0);
			p[0] = -p[1];
		}

		Segment(Vector<N, T> const& p0, Vector<N, T> const& p1)
			:
			p{ p0, p1 }
		{
		}

		Segment(std::array<Vector<N, T>, 2> const& inP)
			:
			p(inP)
		{
		}

		Segment(Vector<N, T> const& center, Vector<N, T> const& direction, T extent)
		{
			SetCenteredForm(center, direction, extent);
		}

		// Manipulation via the centered form.
		void SetCenteredForm(Vector<N, T> const& center,
			Vector<N, T> const& direction, T extent)
		{
			p[0] = center - extent * direction;
			p[1] = center + extent * direction;
		}

		void GetCenteredForm(Vector<N, T>& center,
			Vector<N, T>& direction, T& extent) const
		{
			center = (T)0.5 * (p[0] + p[1]);
			direction = p[1] - p[0];
			extent = (T)0.5 * Normalize(direction);
		}

		
		std::array<Vector<N, T>, 2> p;
		T float_parametr;

	public:
		// Comparisons to support sorted containers.
		bool operator==(Segment const& segment) const
		{
			return p == segment.p;
		}

		bool operator!=(Segment const& segment) const
		{
			return p != segment.p;
		}

		bool operator< (Segment const& segment) const
		{
			return p < segment.p;
		}

		bool operator<=(Segment const& segment) const
		{
			return p <= segment.p;
		}

		bool operator> (Segment const& segment) const
		{
			return p > segment.p;
		}

		bool operator>=(Segment const& segment) const
		{
			return p >= segment.p;
		}
	};

	template<typename T>
	class Triangle
	{     
	    
	public:
		Triangle(Vector<3,T> v0, Vector<3,T> v1, Vector<3,T> v2)
		{
			triangle_data[0] = v0;
			triangle_data[1] = v1;
			triangle_data[2] = v2;

			Vector<3,T> edge1 = v1 - v0;
			Vector<3,T> edge2 = v2 - v0;

			normal =Cross(edge1, edge2);
			Normalize(normal);
		}

		Triangle(Vector<3, T> vec_triangle[3])
		{
			triangle_data[0] = vec_triangle[0];
			triangle_data[1] = vec_triangle[1];
			triangle_data[2] = vec_triangle[2];
		}

		Triangle( const std::array<T, 3>&v0, const std::array<T,3>& v1, const std::array<T,3>& v2)
		{
			triangle_data[0] = v0;
			triangle_data[1] = v1;
			triangle_data[2] = v2;
		}


		Triangle(const Triangle& poly)
		{
			triangle_data[0] = poly.GetVertex(0);
			triangle_data[1] = poly.GetVertex(1);
			triangle_data[2] = poly.GetVertex(2);
			normal = poly.GetNormal();
		}

		Triangle& operator=(const Triangle& poly)
		{
			triangle_data[0] = poly.GetVertex(0);
			triangle_data[1] = poly.GetVertex(1);
			triangle_data[2] = poly.GetVertex(2);
			normal = poly.GetNormal();
			return *this;
		}
				
		Triangle() = default;

		inline Vector<3,T> GetNormal() const
		{
			return normal;
		}
		inline std::array<Vector<3,T>,3> GetArray() const
		{
	   
		   return triangle_data;
		}

		inline Vector<3,T> GetVertex(int vert) const
		{
			return triangle_data[vert];
		}

		inline void SetVertex(int pos, Vector<3,T>& point)
		{
			triangle_data[pos] = point;
		}

	   
	   std::array<Vector<3,T>,3>triangle_data;
	   Vector<3,T>normal;

	};

	template <typename T>
	auto readTrianglesFromFile(const std::string& filename)
	{
		std::vector<Triangle<T>> triangles; // To store the parsed triangles
		std::ifstream file(filename);   // Open the file for reading

		if (!file.is_open()) {
			std::cerr << "Error: Could not open file " << filename << "\n";
			return triangles; // Return an empty vector if the file cannot be opened
		}

		std::string line;
		while (std::getline(file, line)) {
			// Skip empty lines or lines starting with '#'
			if (line.empty() || line[0] == '#') {
				continue;
			}

			std::istringstream iss(line);
			
			std::array<T,3>local_arr{0,0,0};
			std::array<T,3>local_arr_2{0,0,0};
			std::array<T,3>local_arr_3{0,0,0};

			// Read 9 values for the 3 vertices of the triangle
			if (!(iss >> local_arr[0] >>local_arr[1] >> local_arr[2] >>
				local_arr_2[0] >> local_arr_2[1] >> local_arr_2[2] >>
				local_arr_3[0] >> local_arr[1] >>local_arr_3[2])) {
				std::cerr << "Error: Invalid line format: " << line << "\n";
				continue; // Skip invalid lines
			}
         

		Triangle<T>triangle(local_arr,local_arr_2,local_arr_3);
			triangles.push_back(triangle); // Add the parsed triangle to the list
		}

		file.close(); // Close the file after reading
		return triangles;
	}


	template <int32_t N, typename T>
	class GeomCore
	{
	   public:

	   GeomCore()=default;

	   struct Result
	   {
		   Result()
			   :
			   distance(static_cast<T>(0)),
			   sqrDistance(static_cast<T>(0)),
			   parameter{ static_cast<T>(0), static_cast<T>(0) },
			   closest{ Vector<N, T>::Zero(), Vector<N, T>::Zero() }
		   {
		   }

		   T distance, sqrDistance;
		   std::array<T, 2> parameter;
		   std::array<Vector<N, T>, 2> closest;
	   };


	   double ProjectPointOnSegment(const Segment<3, double>& segment,const Vector<3, double>& point)
		{
		   
           Vector<3,double> dist_segment=segment.p[1]-segment.p[0];
		   Vector<3,double> point_segment_dist=point-segment.p[0];
		   
		   Vector<3,double> a = Cross(point_segment_dist, segment.p[1]);
		   return (std::sqrt(DotProduct(a, a) / DotProduct(segment.p[1], segment.p[1])));
  
	   }

	   double DistanceSegmentPricise(Segment<3, double>& segment_1, Segment<3, double>& segment_2)
	   {
	       
		   auto Dot_1=DotProduct(segment_1.p[1],segment_1.p[1]);
		   auto Dot_2 = DotProduct(segment_2.p[1], segment_2.p[1]);
		   auto Dot_3=DotProduct(segment_1.p[1],segment_2.p[1]);
		   auto seg_dif=segment_2.p[0]-segment_1.p[0];

	   }

      
	   double DistanceSegment(Segment<3, double>& segment_1, Segment<3, double>& segment_2)
	   {
	       
		   auto origin_diff_1=segment_2.p[1]-segment_1.p[0];
		   auto origin_diff_2 = segment_1.p[1] - segment_2.p[0];
		   auto segment_1_diff=segment_1.p[1]-segment_1.p[0];
		   auto segment_2_diff = segment_2.p[1] - segment_2.p[0];
		   		   
		   auto Cross_1=Cross(origin_diff_1,origin_diff_2);
		   		   		   
		   auto dot_1 = DotProduct(origin_diff_1, Cross_1)/Lenght(Cross_1);
		   auto dot_2=DotProduct(origin_diff_2,Cross_1)/Lenght(Cross_1);
		   
		   return Lenght(segment_2.p[1] * dot_2 - segment_1.p[1] * dot_1);
		              
		   auto uit=DotProduct((segment_2.p[0]-segment_1.p[0]),Cross(segment_1.p[1],segment_2.p[1])/Lenght(Cross(segment_1.p[1],segment_2.p[1])));

		   /*auto cross_dist_vector=Cross(segment_1.p[1],segment_2.p[1]);

		   auto cross_dest_1=Cross(origin_diff,segment_2.p[1]);
		   auto  cross_dest_2 = Cross(origin_diff, segment_1.p[1]);

		   double t1=DotProduct(cross_dest_1,cross_dist_vector)/std::sqrt(DotProduct(cross_dist_vector,cross_dist_vector));
		   double t2=DotProduct(cross_dest_2,cross_dist_vector)/std::sqrt(DotProduct(cross_dist_vector,cross_dist_vector));


		   return Lenght(segment_2.p[1] * t2 - segment_1.p[1] * t1);*/
		   		  
			auto dp = segment_2.p[0] - segment_1.p[0];
			auto v12 = DotProduct(segment_1.p[1], segment_1.p[1]);
			auto v22 = DotProduct(segment_2.p[1], segment_2.p[1]);
			auto v1v2 = DotProduct(segment_1.p[1], segment_2.p[1]);
			auto det = v1v2 * v1v2 - v12 * v22;

			if (std::fabs(det) > FLT_MIN)
				det = 1.0F / det;
			double dpvl = DotProduct(dp, segment_1.p[1]);
			double dpv2 = DotProduct(dp, segment_2.p[1]);
			double tl = (v1v2 * dpv2 - v22 * dpvl) * det;
			double t2 = (v12 * dpv2 - v1v2 * dpvl) * det;
			return std::sqrt(DotProduct((dp + (segment_2.p[1] * t2) - (segment_1.p[1] * tl)), (dp + (segment_2.p[1] * t2) - (segment_1.p[1] * tl))));

	  }

	   Result operator()(Segment<N,T>&segment_1,
		   Segment<N, T>& segment_2)
	 { 

	   Vector<N, T> P1mP0 = segment_1.p[1]-segment_1.p[0];
	   Vector<N, T> Q1mQ0 = segment_2.p[1]-segment_2.p[0];
	   Vector<N, T> P0mQ0 = segment_1.p[0] - segment_2.p[0];
	   T a = DotProduct(P1mP0, P1mP0);
	   T b = DotProduct(P1mP0, Q1mQ0);
	   T c = DotProduct(Q1mQ0, Q1mQ0);
	   T d = DotProduct(P1mP0, P0mQ0);
	   T e = DotProduct(Q1mQ0, P0mQ0);
	   T det = a * c - b * b;
	   T s, t, nd, bmd, bte, ctd, bpe, ate, btd;

	   T const zero = static_cast<T>(0);
	   T const one = static_cast<T>(1);
	   if (det > zero)
	   {
		   bte = b * e;
		   ctd = c * d;
		   if (bte <= ctd)  // s <= 0
		   {
			   s = zero;
			   if (e <= zero)  // t <= 0
			   {
				   // region 6
				   t = zero;
				   nd = -d;
				   if (nd >= a)
				   {
					   s = one;
				   }
				   else if (nd > zero)
				   {
					   s = nd / a;
				   }
				   // else: s is already zero
			   }
			   else if (e < c)  // 0 < t < 1
			   {
				   // region 5
				   t = e / c;
			   }
			   else  // t >= 1
			   {
				   // region 4
				   t = one;
				   bmd = b - d;
				   if (bmd >= a)
				   {
					   s = one;
				   }
				   else if (bmd > zero)
				   {
					   s = bmd / a;
				   }
				   // else:  s is already zero
			   }
		   }
		   else  // s > 0
		   {
			   s = bte - ctd;
			   if (s >= det)  // s >= 1
			   {
				   // s = 1
				   s = one;
				   bpe = b + e;
				   if (bpe <= zero)  // t <= 0
				   {
					   // region 8
					   t = zero;
					   nd = -d;
					   if (nd <= zero)
					   {
						   s = zero;
					   }
					   else if (nd < a)
					   {
						   s = nd / a;
					   }
					   // else: s is already one
				   }
				   else if (bpe < c)  // 0 < t < 1
				   {
					   // region 1
					   t = bpe / c;
				   }
				   else  // t >= 1
				   {
					   // region 2
					   t = one;
					   bmd = b - d;
					   if (bmd <= zero)
					   {
						   s = zero;
					   }
					   else if (bmd < a)
					   {
						   s = bmd / a;
					   }
					   // else:  s is already one
				   }
			   }
			   else  // 0 < s < 1
			   {
				   ate = a * e;
				   btd = b * d;
				   if (ate <= btd)  // t <= 0
				   {
					   // region 7
					   t = zero;
					   nd = -d;
					   if (nd <= zero)
					   {
						   s = zero;
					   }
					   else if (nd >= a)
					   {
						   s = one;
					   }
					   else
					   {
						   s = nd / a;
					   }
				   }
				   else  // t > 0
				   {
					   t = ate - btd;
					   if (t >= det)  // t >= 1
					   {
						   // region 3
						   t = one;
						   bmd = b - d;
						   if (bmd <= zero)
						   {
							   s = zero;
						   }
						   else if (bmd >= a)
						   {
							   s = one;
						   }
						   else
						   {
							   s = bmd / a;
						   }
					   }
					   else  // 0 < t < 1
					   {
						   // region 0
						   s /= det;
						   t /= det;
					   }
				   }
			   }
		   }
	   }
	   else
	   {
		   // The segments are parallel. The quadratic factors to
		   //   R(s,t) = a*(s-(b/a)*t)^2 + 2*d*(s - (b/a)*t) + f
		   // where a*c = b^2, e = b*d/a, f = |P0-Q0|^2, and b is not
		   // zero. R is constant along lines of the form s-(b/a)*t = k
		   // and its occurs on the line a*s - b*t + d = 0. This line
		   // must intersect both the s-axis and the t-axis because 'a'
		   // and 'b' are not zero. Because of parallelism, the line is
		   // also represented by -b*s + c*t - e = 0.
		   //
		   // The code determines an edge of the domain [0,1]^2 that
		   // intersects the minimum line, or if none of the edges
		   // intersect, it determines the closest corner to the minimum
		   // line. The conditionals are designed to test first for
		   // intersection with the t-axis (s = 0) using
		   // -b*s + c*t - e = 0 and then with the s-axis (t = 0) using
		   // a*s - b*t + d = 0.

		   // When s = 0, solve c*t - e = 0 (t = e/c).
		   if (e <= zero)  // t <= 0
		   {
			   // Now solve a*s - b*t + d = 0 for t = 0 (s = -d/a).
			   t = zero;
			   nd = -d;
			   if (nd <= zero)  // s <= 0
			   {
				   // region 6
				   s = zero;
			   }
			   else if (nd >= a)  // s >= 1
			   {
				   // region 8
				   s = one;
			   }
			   else  // 0 < s < 1
			   {
				   // region 7
				   s = nd / a;
			   }
		   }
		   else if (e >= c)  // t >= 1
		   {
			   // Now solve a*s - b*t + d = 0 for t = 1 (s = (b-d)/a).
			   t = one;
			   bmd = b - d;
			   if (bmd <= zero)  // s <= 0
			   {
				   // region 4
				   s = zero;
			   }
			   else if (bmd >= a)  // s >= 1
			   {
				   // region 2
				   s = one;
			   }
			   else  // 0 < s < 1
			   {
				   // region 3
				   s = bmd / a;
			   }
		   }
		   else  // 0 < t < 1
		   {
			   // The point (0,e/c) is on the line and domain, so we have
			   // one point at which R is a minimum.
			   s = zero;
			   t = e / c;
		   }
	   }

	   Result result{};
	   result.parameter[0] = s;
	   result.parameter[1] = t;
	   result.closest[0] = segment_1.p[0] + s * P1mP0;
	   result.closest[1] = segment_2.p[0] + t * Q1mQ0;
	   Vector<N, T> diff = result.closest[0] - result.closest[1];
	   result.sqrDistance = DotProduct(diff, diff);
	   result.distance = std::sqrt(result.sqrDistance);
	   return result;
 }
   
template<typename T>  
bool Intersects(std::array<Vector<3, double>, 3> U,
	std::array<Vector<3, double>, 3>V, Vector<3,T> segment[2]) const
{
	// Compute the plane normal for triangle U.
	Vector<3,T> edge1 = U[1] - U[0];
	Vector<3,T> edge2 = U[2] - U[0];
	Vector<3,T> normal = UnitCross(edge1, edge2);

	// Test whether the edges of triangle V transversely intersect the
	// plane of triangle U.
	double d[3];
	int32_t positive = 0, negative = 0, zero = 0;
	for (int32_t i = 0; i < 3; ++i)
	{
		d[i] = DotProduct(normal, V[i] - U[0]);
		if (d[i] > 0.0f)
		{
			++positive;
		}
		else if (d[i] < 0.0f)
		{
			++negative;
		}
		else
		{
			++zero;
		}
	}
	// positive + negative + zero == 3
	//We will use linear interpolation to find segmets c=a+t*b,t-interpolation factor,d will be scalar factor here
	if (positive > 0 && negative > 0)
	{
		if (positive == 2)  // and negative == 1
		{
			if (d[0] < 0.0f)
			{
				segment[0] = (d[1] * V[0] - d[0] * V[1]) / (d[1] - d[0]);
				segment[1] = (d[2] * V[0] - d[0] * V[2]) / (d[2] - d[0]);
			}
			else if (d[1] < 0.0f)
			{
				segment[0] = (d[0] * V[1] - d[1] * V[0]) / (d[0] - d[1]);
				segment[1] = (d[2] * V[1] - d[1] * V[2]) / (d[2] - d[1]);
			}
			else  // d[2] < 0.0f
			{
				segment[0] = (d[0] * V[2] - d[2] * V[0]) / (d[0] - d[2]);
				segment[1] = (d[1] * V[2] - d[2] * V[1]) / (d[1] - d[2]);
			}
		}
		else if (negative == 2)  // and positive == 1
		{
			if (d[0] > 0.0f)
			{
				segment[0] = (d[1] * V[0] - d[0] * V[1]) / (d[1] - d[0]);
				segment[1] = (d[2] * V[0] - d[0] * V[2]) / (d[2] - d[0]);
			}
			else if (d[1] > 0.0f)
			{
				segment[0] = (d[0] * V[1] - d[1] * V[0]) / (d[0] - d[1]);
				segment[1] = (d[2] * V[1] - d[1] * V[2]) / (d[2] - d[1]);
			}
			else  // d[2] > 0.0f
			{
				segment[0] = (d[0] * V[2] - d[2] * V[0]) / (d[0] - d[2]);
				segment[1] = (d[1] * V[2] - d[2] * V[1]) / (d[1] - d[2]);
			}
		}
		else  // positive == 1, negative == 1, zero == 1
		{
			if (d[0] == 0.0f)
			{
				segment[0] = V[0];
				segment[1] = (d[2] * V[1] - d[1] * V[2]) / (d[2] - d[1]);
			}
			else if (d[1] == 0.0f)
			{
				segment[0] = V[1];
				segment[1] = (d[0] * V[2] - d[2] * V[0]) / (d[0] - d[2]);
			}
			else  // d[2] == 0.0f
			{
				segment[0] = V[2];
				segment[1] = (d[1] * V[0] - d[0] * V[1]) / (d[1] - d[0]);
			}
		}
		return true;
	}

	// Triangle V does not transversely intersect triangle U, although it is
	// possible a vertex or edge of V is just touching U.  In this case, we
	// do not call this an intersection.
	return false;
}

bool TriangleIntersection ( std::array<Vector<3,double>,3> tri_1,std::array<Vector<3,double>,3> tri_2) const
{
	Vector<3,double> S0[2], S1[2];
	if (Intersects(tri_1, tri_2, S0) && Intersects(tri_1,tri_2, S1))
	{
		// Theoretically, the segments lie on the same line.  A direction D
		// of the line is the Cross(NormalOf(U),NormalOf(V)).  We choose the
		// average A of the segment endpoints as the line origin.
		Vector<3,T> uNormal = Cross(tri_1[1] - tri_1[0], tri_1[2] - tri_1[0]);
		Vector<3,T> vNormal = Cross(tri_2[1] - tri_2[0], tri_2[2] - tri_2[0]);
		Vector<3,T> D = UnitCross(uNormal, vNormal);
		Vector<3, double> A= 0.25 * (S0[0] + S0[1] + S1[0] + S1[1]);
		
		// Each segment endpoint is of the form A + t*D.  Compute the
		// t-values to obtain I0 = [t0min,t0max] for S0 and I1 = [t1min,t1max]
		// for S1.  The segments intersect when I0 overlaps I1.  Although this
		// application acts as a "test intersection" query, in fact the
		// construction here is a "find intersection" query.
		float t00 = DotProduct(D, S0[0] - A), t01 = DotProduct(D, S0[1] - A);
		float t10 = DotProduct(D, S1[0] - A), t11 = DotProduct(D, S1[1] - A);
		auto I0 = std::minmax(t00, t01);
		auto I1 = std::minmax(t10, t11);
		return (I0.second > I1.first && I0.first < I1.second);
	}
	return false;
}


};

}







