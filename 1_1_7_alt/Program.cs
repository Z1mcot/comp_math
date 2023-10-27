using System.Numerics;

namespace _1_1_7_alt
{
    internal class Program
    {
        static double[,] SolveMatrixEquation(double[,] A, double[,] F)
        {
            int rowsA = A.GetLength(0);
            int columnsA = A.GetLength(1);
            
            int columnsF = F.GetLength(1);

            // Создаем матрицу X для хранения решения
            double[,] X = new double[columnsA, columnsF];

            // Копируем матрицу B в матрицу X
            for (int i = 0; i < columnsA; i++)
            {
                for (int j = 0; j < columnsF; j++)
                {
                    X[i, j] = F[i, j];
                }
            }

            // Выполняем метод поворота
            for (int k = 0; k < columnsA; k++)
            {
                for (int j = k + 1; j < columnsA; j++)
                {
                    // Вычисляем коэффициенты поворота c и s
                    double r = Math.Sqrt(A[k, k] * A[k, k] + A[j, k] * A[j, k]);
                    double c = A[k, k] / r;
                    double s = A[j, k] / r;

                    // Применяем поворот к матрицам A и X
                    for (int i = 0; i < rowsA; i++)
                    {
                        double tempA = A[k, i];
                        A[k, i] = c * tempA + s * A[j, i];
                        A[j, i] = -s * tempA + c * A[j, i];
                    }

                    for (int i = 0; i < columnsA; i++)
                    {
                        double tempX = X[k, i];
                        X[k, i] = c * tempX + s * X[j, i];
                        X[j, i] = -s * tempX + c * X[j, i];
                    }
                }
            }

            // Решаем верхнетреугольную матрицу
            for (int k = columnsA - 1; k >= 0; k--)
            {
                for (int j = 0; j < columnsF; j++)
                {
                    X[k, j] /= A[k, k];
                    for (int i = 0; i < k; i++)
                    {
                        X[i, j] -= A[i, k] * X[k, j];
                    }
                }
            }

            return X;
        }

        static double[,] MultiplyMatrices(double[,] A, double[,] B)
        {
            int rowsA = A.GetLength(0);
            int columnsA = A.GetLength(1);

            int columnsB = B.GetLength(1);

            // Создаем новую матрицу для хранения результата умножения
            double[,] result = new double[rowsA, columnsB];

            // Выполняем умножение матриц
            for (int i = 0; i < rowsA; i++)
            {
                for (int j = 0; j < columnsB; j++)
                {
                    double sum = 0;
                    for (int k = 0; k < columnsA; k++)
                    {
                        sum += A[i, k] * B[k, j];
                    }
                    result[i, j] = sum;
                }
            }

            return result;
        }

        static double[,] SubtractMatrices(double[,] A, double[,] B)
        {
            int rowsA = A.GetLength(0);
            int columnsA = A.GetLength(1);
            
            // Создаем новую матрицу для хранения результата вычитания
            double[,] result = new double[rowsA, columnsA];

            // Выполняем вычитание матриц
            for (int i = 0; i < rowsA; i++)
            {
                for (int j = 0; j < columnsA; j++)
                {
                    result[i, j] = A[i, j] - B[i, j];
                }
            }

            return result;
        }

        static double[,] CalcResidual(double[,] A, double[,] X, double[,] F) 
            => SubtractMatrices(F, MultiplyMatrices(A, X));

        static double[,] RoundMatrix(double[,] matrix, int decimalPlaces)
        {
            int rows = matrix.GetLength(0);
            int columns = matrix.GetLength(1);

            // Создаем новую матрицу для хранения округленных значений
            double[,] result = new double[rows, columns];

            // Округляем каждый элемент матрицы
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    result[i, j] = Math.Round(matrix[i, j], decimalPlaces);
                }
            }

            return result;
        }

        static void Main()
        {
            // Задаем матрицы A и F
            double[,] A = { 
                { 1.0, 0.42, 0.54, 0.66 }, 
                { 0.42, 1.0, 0.32, 0.44 }, 
                { 0.54, 0.32, 1.0, 0.22 }, 
                { 0.66, 0.44, 0.22, 1.0 } 
            };

            double[,] F = { 
                { 1.0, 0.0, 0.0, 0.0 }, 
                { 0.0, 1.0, 0.0, 0.0 }, 
                { 0.0, 0.0, 1.0, 0.0 }, 
                { 0.0, 0.0, 0.0, 1.0 } 
            };

            double[,] X = RoundMatrix(SolveMatrixEquation(A.Copy(), F.Copy()), 8);

            int columnsA = A.GetLength(1);
            int columnF = F.GetLength(1);

            // Выводим решение
            Console.WriteLine("Решение:");
            for (int i = 0; i < columnsA; i++)
            {
                for (int j = 0; j < columnF; j++)
                {
                    Console.Write($"{X[i, j]}\t");
                }
                Console.WriteLine();
            }

            double[,] residual = RoundMatrix(CalcResidual(A, X, F), 8);
            // Выводим решение
            Console.WriteLine("Матрица невязки:");
            for (int i = 0; i < columnsA; i++)
            {
                for (int j = 0; j < columnF; j++)
                {
                    Console.Write($"{residual[i, j]}\t");
                }
                Console.WriteLine();
            }
        }
    }

    public static class ArrayExtensions
    {
        public static T[,] Copy<T>(this T[,] arr)
        {
            return (T[,])arr.Clone();
        }
    }
}