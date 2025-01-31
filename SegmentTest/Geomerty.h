#pragma once
#include<array>
#include<iostream>
#include<tuple>


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
  std::array<T,N>Vector_Array;

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

	template <int32_t N, typename T>
	class GeomCore
	{
	   public:

	   GeomCore()=default;

	   double ProjectPointOnSegment(const Segment<3, double>& segment,const Vector<3, double>& point)
		{
		   
           Vector<3,double> dist_segment=segment.p[1]-segment.p[0];
		   Vector<3,double> point_segment_dist=point-segment.p[0];
		   
		   Vector<3,double> a = Cross(point_segment_dist, segment.p[1]);
		   return (std::sqrt(DotProduct(a, a) / DotProduct(segment.p[1], segment.p[1])));
  
	   }
      
	   double DistanceSegment(Segment<3, double>& segment_1, Segment<3, double>& segment_2)
	   {
	       
		   /*Vector<3,double> t1_crossed=Cross(segment_2.p[0]-segment_1.p[0],segment_2.p[1]);
		   Vector<3, double> t2_crossed = Cross(segment_2.p[0] - segment_1.p[0], segment_1.p[1]);
		   Vector<3,double> cross_dist_vectors=Cross(segment_1.p[1],segment_2.p[1]);

		   double t1=DotProduct(t1_crossed,cross_dist_vectors)/std::sqrt(DotProduct(cross_dist_vectors,cross_dist_vectors));

		   double t2 = DotProduct(t2_crossed, cross_dist_vectors) / std::sqrt(DotProduct(cross_dist_vectors, cross_dist_vectors));

		   auto vec2_dir=segment_2.p[0] + t2 * segment_2.p[1];
		   auto vec1_dir = segment_1.p[0] + t1 * segment_1.p[1];

		   double distance=std::sqrt(DotProduct(vec2_dir-vec1_dir, vec2_dir - vec1_dir));

		   return distance;*/

		  
		   auto dp = segment_2.p[0] - segment_1.p[0];
		   auto v12 = DotProduct(segment_1.p[1], segment_1.p[1]);
		   auto v22 = DotProduct(segment_2.p[1], segment_2.p[1]);
		   auto v1v2 = DotProduct(segment_1.p[1], segment_2.p[1]);
		   auto det = v1v2 * v1v2 - v12 * v22;

		   if (std::fabs(det) > FLT_MIN)
			 det = 1.0F / det;
		   double dpvl = DotProduct(dp, segment_1.p[1]);
		   double dpv2 = DotProduct(dp, segment_2.p[1]);
		   double tl=(v1v2 * dpv2 - v22 * dpvl)* det;
		   double t2 = (v12 * dpv2 - v1v2 * dpvl) * det;
		   return std::sqrt(DotProduct((dp + (segment_2.p[1] * t2) - (segment_1.p[1] * tl)), (dp + (segment_2.p[1] * t2) - (segment_1.p[1] * tl))));

	  }


 };


};







