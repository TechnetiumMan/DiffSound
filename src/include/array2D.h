#pragma once
#include "array.h"
namespace ModalSound
{
	template <typename T>
	class GArr2D;
	template <typename T>
	class CArr2D;

	template <typename T>
	class GArr2D
	{
	public:
		GArr<T> data;
		int rows, cols;
		void resize(int rows, int cols)
		{
			this->rows = rows;
			this->cols = cols;
			this->data.resize(rows * cols);
		}
		GArr2D() {}
		GArr2D(int rows, int cols)
		{
			this->resize(rows, cols);
		}

		GArr2D(T *data, int rows, int cols)
		{
			this->rows = rows;
			this->cols = cols;
			this->data = GArr<T>(data, rows * cols);
		}
		GArr2D(CArr2D<T> &A)
		{
			this->assign(A);
		}
		void assign(CArr2D<T> &A)
		{
			this->rows = A.rows;
			this->cols = A.cols;
			data.assign(A.data);
		}

		void copy_from(const GArr2D<T> &A)
		{
			data.copy_from(A.data);
		}

		void clear()
		{
			data.clear();
		}
		void reset()
		{
			data.reset();
		}
		void reset_minus_one()
		{
			data.reset_minus_one();
		}

		~GArr2D(){};

		GPU_FUNC inline const T &operator()(const uint i, const uint j) const
		{
			return data[i * cols + j];
		}

		GPU_FUNC inline T &operator()(const uint i, const uint j)
		{
			return data[i * cols + j];
		}

		CGPU_FUNC inline int index(const uint i, const uint j) const
		{
			return i * cols + j;
		}

		inline GArr<T> operator[](unsigned int id)
		{
			return GArr<T>(begin() + id * cols, cols);
		}

		inline const GArr<T> operator[](unsigned int id) const
		{
			return GArr<T>(begin() + id * cols, cols);
		}

		CGPU_FUNC inline const T *begin() const { return data.begin(); }
		CGPU_FUNC inline T *begin() { return data.begin(); }

		inline CArr2D<T> cpu()
		{
			return CArr2D<T>(*this);
		}

		friend std::ostream &operator<<(std::ostream &os, GArr2D<T> &mat)
		{
			os << mat.cpu();
			return os;
		}
	};

	template <typename T>
	class CArr2D
	{
	public:
		CArr<T> data;
		int rows, cols;
		CArr2D() {}
		void resize(int rows, int cols)
		{
			this->rows = rows;
			this->cols = cols;
			this->data.resize(rows * cols);
		}
		CArr2D(int rows, int cols)
		{
			this->resize(rows, cols);
		}

		CArr2D(T *data, int rows, int cols)
		{
			this->rows = rows;
			this->cols = cols;
			this->data.clear();
			this->data = CArr<T>(data, rows * cols);
		}

		CArr2D(GArr2D<T> &A)
		{
			this->assign(A);
		}
		void assign(GArr2D<T> &A)
		{
			this->rows = A.rows;
			this->cols = A.cols;
			data.assign(A.data);
		}
		void clear()
		{
			data.clear();
		}

		void reset()
		{
			data.reset();
		}
		void reset_minus_one()
		{
			data.reset_minus_one();
		}

		inline const T &operator()(const uint i, const uint j) const
		{
			return data[i * cols + j];
		}

		inline T &operator()(const uint i, const uint j)
		{
			return data[i * cols + j];
		}

		inline CArr<T> operator[](unsigned int id)
		{
			return CArr<T>(begin() + id * cols, cols);
		}

		inline const CArr<T> operator[](unsigned int id) const
		{
			return CArr<T>(begin() + id * cols, cols);
		}

		inline int index(const uint i, const uint j) const
		{
			return i * cols + j;
		}

		friend std::ostream &operator<<(std::ostream &os, const CArr2D<T> &mat)
		{
			for (int i = 0; i < mat.rows; i++)
			{
				for (int j = 0; j < mat.cols; j++)
				{
					os << mat(i, j) << " ";
				}
				os << std::endl;
			}
			return os;
		}

		inline GArr2D<T> gpu()
		{
			return GArr2D<T>(*this);
		}

		inline CArr<T> column(int i)
		{
			CArr<T> col(rows);
			for (int j = 0; j < rows; j++)
			{
				col[j] = data[j * cols + i];
			}
			return col;
		}

		inline CArr2D<T> transpose()
		{
			CArr2D<T> result(cols, rows);
			for (int i = 0; i < rows; i++)
			{
				for (int j = 0; j < cols; j++)
				{
					result(j, i) = data[i * cols + j];
				}
			}
			return result;
		}

		inline const T *begin() const { return data.begin(); }
		inline T *begin() { return data.begin(); }
	};
}