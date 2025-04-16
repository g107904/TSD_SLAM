#include "c_PointLineBA.h"
#include "c_PL_AdaLam.hpp"
#include <Eigen/Core>
#include <Python.h>
#include <ndarrayobject.h>
#include <iostream>

class c_initSystem
{
public:
	std::vector<cv::Mat> m_file_img;
	void receive_frame(cv::Mat img)
	{
		m_file_img.push_back(img);
	}

	c_initSystem(Eigen::Matrix3f& K)
	{
		m_PointLineBA = new c_PointLineBA();
		m_K = K;
	}

	void initial(Eigen::Matrix3f& K,cv::Mat img)
	{
		receive_frame(img);
		do_diamond_line();
		do_Point_Line_BA();
		//K = (m_diamond_K+m_PointLineBA->m_K)/2;
	}

	void do_diamond_line()
	{
		m_diamond_K = Eigen::Matrix3f::Zero();
		Py_Initialize();
		if (!Py_IsInitialized()) {
			std::cout << "python init fail" << std::endl;
			return;
		}

		PyObject* pModule = PyImport_ImportModule("c_DiamondLine");

		PyObject* pFunc = PyObject_GetAttrString(pModule, "receive_K");

		PyObject* pArgs = PyTuple_New(4);
		PyTuple_SetItem(pArgs, 0, Py_BuildValue("s", m_K(0,0)));
		PyTuple_SetItem(pArgs, 1, Py_BuildValue("s", m_K(1, 1)));
		PyTuple_SetItem(pArgs, 2, Py_BuildValue("s", m_K(0, 2)));
		PyTuple_SetItem(pArgs, 3, Py_BuildValue("s", m_K(1, 2)));
		PyObject* pRetValue = PyObject_CallObject(pFunc, pArgs);

		for (int i = 0; i < m_file_img.size(); i++)
		{
			pFunc = PyObject_GetAttrString(pModule, "receive_img");
			pArgs = PyTuple_New(1);
			npy_intp dims[] = { m_file_img[i].rows, m_file_img[i].cols, 3 };
			PyObject* pValue = PyArray_SimpleNewFromData(3, dims, NPY_UINT8, m_file_img[i].data);
			PyTuple_SetItem(pArgs, 0, pValue);
			pRetValue = PyObject_CallObject(pFunc, pArgs);
		}
		
		pFunc = PyObject_GetAttrString(pModule, "do_diamond_line");
		pRetValue = PyObject_CallObject(pFunc, pArgs);
		m_diamond_K(0,0) = atof(PyUnicode_AsUTF8(PyTuple_GetItem(pRetValue, 0)));
		m_diamond_K(1,1) = atof(PyUnicode_AsUTF8(PyTuple_GetItem(pRetValue, 1)));
		m_diamond_K(0, 2) = atof(PyUnicode_AsUTF8(PyTuple_GetItem(pRetValue, 2)));
		m_diamond_K(1, 2) = atof(PyUnicode_AsUTF8(PyTuple_GetItem(pRetValue, 3)));
		m_diamond_K(2, 2) = 1;
	}

	void do_Point_Line_BA()
	{
		m_PointLineBA->set_K(m_K);
		m_PointLineBA->receive_frame(m_file_img.back());
		m_PointLineBA->do_BA();
	}


	c_PointLineBA* m_PointLineBA;
	c_PL_AdaLAM* m_PL_AdaLAM;
	Eigen::Matrix3f m_K;
	Eigen::Matrix3f m_diamond_K;

};