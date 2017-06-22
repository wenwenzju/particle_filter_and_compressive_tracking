// --seq_path D:/data_seq/Human7 --seq_name Human7 --start_frame 1 --end_frame 250 --nz 4 --initx 110 --inity 111 --initw 37 --inith 116
// --seq_path D:/data_seq/Blurface --seq_name Blurface --start_frame 1 --end_frame 493 --nz 4 --initx 246 --inity 226 --initw 94 --inith 114
 
 #include "CompressiveTracker.h"
 #include "boost/program_options.hpp"
 #include "boost/filesystem.hpp"
 #include "opencv2/opencv.hpp"
 #include "timer.h"
 
 #include <string>
 #include <fstream>
 #include <iomanip>
 
 //#define SAVE_IMAGE
 
 using namespace boost::program_options;
 using namespace std;
 
 int main(int argc, char** argv)
 {
 	using namespace cv;
 	options_description desc("tracking using compressive sensing");
 	desc.add_options()
 		("help,h", "produce help message")
 
 		("seq_path", value<string>(), "sequence path")
 		("seq_name", value<string>(), "sequence name")
 		("start_frame", value<int>()->default_value(1), "start frame number")
 		("end_frame", value<int>(), "end frame number")
 		("nz", value<int>()->default_value(4), "number of zeros")
 		("ext", value<string>()->default_value("jpg"), "image format")
 		("initx", value<int>(), "initial x")
 		("inity", value<int>(), "initial y")
 		("initw", value<int>(), "initial width")
 		("inith", value<int>(), "initial height");
 	variables_map vm;
 
 	string seq_path, seq_name;
 	int start_frame, end_frame, nz;
 	string ext;
 	int initx, inity, initw, inith;
 	try
 	{
 		store(parse_command_line(argc, argv, desc), vm);
 		notify(vm);
 		if (vm.count("help"))
 		{
 			cout << desc << endl;
 			return 1;
 		}
 		if (vm.count("seq_path")){
 			seq_path = vm["seq_path"].as<string>();
 			if (seq_path[seq_path.size()-1] != '\\' && seq_path[seq_path.size()-1] != '/')
 				seq_path += "/";
 		}
 		if (vm.count("seq_name"))
 			seq_name = vm["seq_name"].as<string>();
 		if (vm.count("start_frame"))
 			start_frame = vm["start_frame"].as<int>();
 		if (vm.count("end_frame"))
 			end_frame = vm["end_frame"].as<int>();
 		if (vm.count("nz"))
 			nz = vm["nz"].as<int>();
 		if (vm.count("ext"))
 			ext = vm["ext"].as<string>();
 		if (vm.count("initx"))
 			initx = vm["initx"].as<int>();
 		if (vm.count("inity"))
 			inity = vm["inity"].as<int>();
 		if (vm.count("initw"))
 			initw = vm["initw"].as<int>();
 		if (vm.count("inith"))
 			inith = vm["inith"].as<int>();
 
 		ofstream result_file(seq_name+"_CT.txt");
 		result_file << initx << "	" << inity << "	" << initw << "	" << inith << endl;
 		ofstream fps_file(seq_name+"_CT_FPS.txt");
 		CompressiveTracker tracker;
 
 		cv::Rect initialization(initx, inity, initw, inith);
 		cv::Mat image;
 
 		//read first image
 		stringstream ss;
 		ss << seq_path;
 		ss << setfill('0') << setw(nz) << start_frame;
 		ss << "." << ext;
 		string imgname;
 		ss >> imgname;
 		image = imread(imgname);
 
 		tic
 			tracker.init(image, initialization);
 
 		Rect rect = initialization;
 		for (int i = start_frame+1; i <= end_frame; ++i) {
 
 			stringstream ss;
 			ss << seq_path;
 			ss << setfill('0') << setw(nz) << i;
 			ss << "." << ext;
 			string imgname;
 			ss >> imgname;
 
 			cv::Mat image = cv::imread(imgname);
 
 			tracker.track(image, rect);
 			result_file << rect.x << "	" << rect.y << "	" << rect.width << "	" << rect.height << endl;
 
 			//cv::rectangle(image, rect, cv::Scalar(0, 0, 255), 2);
 			//cv::imshow("result", image);
 			////cout << imgname << endl;
 			//char k = cv::waitKey(2);
 			//if (k=='q') return 0;
 			//if (k=='p') cv::waitKey(0);
 #ifdef SAVE_IMAGE
 			stringstream ss1;
 			ss1 << setfill('0') << setw(nz) << i;
 			ss1 << "." << ext;
 			string in; ss1 >> in;
 			imwrite(in, image);
 #endif
 
 		}
 		double tot_time = toc;
 		cout << "FPS: " << 1.*(end_frame - start_frame + 1) / tot_time << endl;
 		fps_file << 1.*(end_frame - start_frame + 1) / tot_time;
 	}
 	catch (invalid_command_line_syntax&)
 	{
 		cout << "syntax error" << endl;
 		cout << desc << endl;
 	}
 	catch (boost::bad_lexical_cast&)
 	{
 		cout << "lexical error" << endl;
 		cout << desc << endl;
 	}
 	catch (...)
 	{
 		cout << "Error caught!" << endl;
 	}
 
 
 }

//#include <opencv2/opencv.hpp>
//#include <iostream>
//using namespace cv;
//using namespace std;
//
//int main()
//{
//	Mat m1 = 0.5*Mat::ones(1, 5, CV_32FC1);
//	Mat m2 = 0.6*Mat::ones(1, 5, CV_32FC1);
//
//	m1.mul(m2);
//
//	cout << m1 << endl;
//	//cout << m3 << endl;
//}