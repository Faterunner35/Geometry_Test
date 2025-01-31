#pragma once
#include<array>
#include<vector>
template<typename T,uint32_t Column,uint32_t Rows>
class Matrix
{   
   
   public:
 
	Matrix()
	{    
	    ColumnSize=Column;
		RowSize=Rows;
		array_of_numbers.fill(T(0));

   }


 inline	T& GetRow(uint32_t row, uint32_t column) const 
	{
	    
		return &array_of_numbers[(row*RowSize+column)];
	
   }

 void SetRow(uint32_t rowNum, std::vector<T>data)
 {
	 if (RowSize <= data.size())

	 {
		 for (int i = 0; i < ColumnSize; i++)
		 {
			 GetRow(rowNum,i)=data[i];
		 }

	 }

   }


private:
   uint32_t ColumnSize;
   uint32_t RowSize;
   std::array<T,Column*Rows>array_of_numbers;
};