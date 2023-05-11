/*
    This is C# code for the RANSAC linear regression method. This method is
    implemented to prevent the influence of extreme outliers on the results
    of a linear regression method.

    This code was written to be integrated in a product for OptoQuest as a
    part of the Cleveland Clinic.

    Author: Brian Andrews
    Date: 2018
*/

﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace ransac
{
    class Program
    {
        //struct to help organize code and store arrays that will store the sample subsets
        public struct allLists
        {
            public List<double> xn;
            public List<double> yn;
        }

        //function to calculate slope and intercept of line via linear least squares
        static public double[] calcParams(double len, List<double> x, List<double> y)
        {
            double[] param = new double[2];
            double x2sum = 0;
            double xysum = 0;
            //calculate sum of x^2 and xy
            for (int h = 0; h < x.Count; h++)
            {
                x2sum += x[h] * x[h];
                xysum += x[h] * y[h];
            }
            double xsum = x.Sum();
            double ysum = y.Sum();

            param[0] = (1 / ((len * x2sum) - (xsum * xsum))) * ((len * xysum) - (xsum * ysum));
            param[1] = (1 / ((len * x2sum) - (xsum * xsum))) * ((x2sum * ysum) - (xsum * xysum));

            return param;
        }

        static public bool checkModel(double check, double n, double m, double b, double thres, List<double> x, List<double> y)
        {
            double countin = 0;
            double yprime;
            for (int u = 0; u < x.Count; u++)
            {
                yprime = (x[u] * m) + b; //calculate expected y from model
                //Console.WriteLine(Math.Sqrt((yprime - y[u]) * (yprime - y[u])));
                if (Math.Sqrt((yprime - y[u]) * (yprime - y[u])) <= thres) //if error is within threshold, consider point an inlier
                {
                    countin++;
                }
            }
            if((countin/x.Count) >= check) //if final model is approved, write outliers to a file and return true
            {
                return true;
            }
            return false;

        }

        //method used to pick a potential sample to create model
        public static allLists pickSample(Random ran, double n, List<double> x, List<double> y)
        {
            var arrays = new allLists //initiate lists to store subsets
            {
                xn = new List<double>(),
                yn = new List<double>(),
            };

            List<int> index = new List<int>(); //list to store indices of points chose to prevent duplicates

            int dum = 0;
            int anotherfound; //bool variable

            for (int z = 0; z < n; z++) //loops that will choose the subset of data
            {
                anotherfound = 0;
                while (anotherfound == 0) //loop to stop duplicates
                {
                    dum = ran.Next(0, x.Count);
                    for (int c = 0; c < index.Count; c++) //traverse list for duplicates
                    {
                        if (dum == index[c]) //if the index is found
                        {
                            anotherfound = 1;
                        }
                    }
                    if (anotherfound == 0) //if it is not found, break loop and add to lists
                    {
                        break;
                    }
                    if (anotherfound == 1) //if found, need a new random index to work with
                    {
                        anotherfound = 0;
                    }
                }
                //add index and values to the necessary lists
                index.Add(dum);
                arrays.xn.Add(x[dum]);
                arrays.yn.Add(y[dum]);
            }

            return arrays;
        }

        static void Main(string[] args)
        {
            //setting up random number generator
            Random rand = new Random();

            //section for reading the manufactured .txt files
            string[] lines = System.IO.File.ReadAllLines(@"C:\Users\Brian Andrews\OneDrive\OptoQuest\ransac\cyl.txt");

            //list to hold all of the x and y data points
            List<double> x = new List<double>();
            List<double> y = new List<double>();

            foreach(string line in lines)
            {
                string dumb = "";
                string dumb2 = "";
                double num;
                double num1;
                int found = 0;
                for (int i = 0; i < line.Length; i++)
                {
                    if(line[i] == '\t')
                    {
                        found = 1;
                    }
                    if(found == 0 && line[i] != '\t')
                    {
                        dumb += line[i];
                    }
                    if (found == 1 && line[i] != '\t')
                    {
                        dumb2 += line[i];
                    }
                }
                num = System.Convert.ToDouble(dumb);
                num1 = System.Convert.ToDouble(dumb2);
                x.Add(num);
                y.Add(num1);
            }

            /*
             * Start RANSAC method
             * Will be fitting a linear model
             * The following values are the parameters that govern the algorithm:
             * n - Establish a minimum number of data points to take from data set to create fit
             * k - Establish maximum number of allowed iterations
             * eps - Establish a threshold for a data point being an "inlier"
             * goodif - Establish a number of inliers necessary to accept the proposed model
            */

            double n = Math.Ceiling(.5 * x.Count); //50% of the data points is needed to create the model
            Console.WriteLine(n);
            double k = 1000;
            double eps = .5;
            double goodif = .6; //75% of the collected data need to satisfy the criterion of model

            //now start the main loop
            double l = 0; //counter variable for main loop
            while(l < k)
            {
                //pick a subset of x and y points
                var sample = pickSample(rand, n, x, y);

                /*
                 * Fit sample of data with Linear Least Squares Method
                */

                double[] parameters = new double[2]; //array for slope and intercept
                parameters = calcParams(n, sample.xn, sample.yn); //calculate those parameters
                //Console.WriteLine(parameters[0] + " " + parameters[1]);

                //start section where the model is checked for the desired accuracy
                if(checkModel(goodif,n,parameters[0],parameters[1],eps,x,y))
                {
                    Console.WriteLine(parameters[0] + " " + parameters[1]);
                    break;
                    Console.WriteLine("after break");
                }

                l++;
            }

            Console.WriteLine("\n Press c or anything really");
            System.Console.ReadKey();
        }
    }
}
