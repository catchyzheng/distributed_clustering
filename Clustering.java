package clustering;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.net.URI;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.LineReader;

public class Clustering {
	private static int centerNum = 5; // 类数
	private static int dim = 2; // 维数
	private static String namePath = "hdfs://192.168.15.128:9000";
	private static String dataPath = "/user/clustering/five_cluster";
	private static String outputPath = "/user/clustering/output";
	private static String centerPath = "/user/clustering/center";
	private static String resultName = "/part-r-00000";
	private static Configuration conf;
	// Mapper 归类每一个点
	public static class ClassifierMapper extends Mapper<Object, Text, IntWritable, Point> {
		private ArrayList<Point> centers; // 中心
		public void setup(Context context) 
				throws IOException, InterruptedException { // 从中心文件中获取当前的中心
			super.setup(context);
			centers = getPoints(centerPath);
		}
		// map 将点归类到最近的中心上
		public void map(Object key, Text value, Context context) 
			throws IOException, InterruptedException {
			double minDist = Double.MAX_VALUE;
			int id = 0;
			Point point = new Point();
			point.readText(value);
			for (int i = 0; i < centerNum; i++) {
				double dist = point.getDistance(centers.get(i));
				if (dist < minDist) {
					minDist = dist;
					id = i;
				}
			}
			context.write(new IntWritable(id), point);
		}
	}
	
	// Combiner 先在节点计算一次中心
	public static class ClusterCombiner extends Reducer<IntWritable, Point, IntWritable, Point> {
		public void reduce(IntWritable key, Iterable<Point> points, Context context)
				throws IOException, InterruptedException {
			Point center = new Point(0);
			int cnt = 0;
			for (Point point : points) {
				center.plus(point);
				++cnt;
			}
			if (cnt > 0) // 获得中心
				center.mul(1.0/cnt);
			context.write(key, center);
		}
	}
	
	// Reducer 将Combiner得到的中心点合并
	public static class CenterReducer extends Reducer<IntWritable, Point, Text, IntWritable> {
		public void reduce(IntWritable key, Iterable<Point> points, Context context)
				throws IOException, InterruptedException {
			Point center = new Point(0);
			int cnt = 0;
			for (Point point : points) {
				center.plus(point);
				++cnt;
			}
			if (cnt > 0) // 获得中心
				center.mul(1.0/cnt);
			context.write(center.toText(), key);
		}
	}
	
	// Reducer 只执行一次，给每个点确定中心点（聚类）
	public static class PointCenterReducer extends Reducer<IntWritable, Point, Text, IntWritable> {
		public void reduce(IntWritable key, Iterable<Point> points, Context context)
				throws IOException, InterruptedException {
			for (Point point : points) {
				context.write(point.toText(), key);
			}
		}
	}
	

	public static void main(String[] args) throws Exception {
		System.setProperty("hadoop.home.dir", "C:\\hadoop-2.6.0\\");
	    conf = new Configuration();
	    conf.set("fs.default.name", "hdfs://192.168.15.128:9000");
	    FileSystem fs = FileSystem.get(URI.create(namePath), conf);
	    JobConf jobConf = new JobConf();
	    jobConf.setJarByClass(Clustering.class);
	    jobConf.setMapOutputKeyClass(IntWritable.class);
	    jobConf.setMapOutputValueClass(Point.class);
	    jobConf.setOutputKeyClass(Text.class);
	    jobConf.setOutputValueClass(IntWritable.class);
	    jobConf.set("fs.default.name", "hdfs://192.168.15.128:9000");
		Point.length = dim; // 设置点的维度
		randomCenters(); // 随机设置中心到output
		while (true) {
			// 拷贝output到center
			FileUtil.copy(fs, new Path(outputPath + resultName), fs, new Path(centerPath), false, true, conf);
			// 删除output
			deletePath(fs, new Path(outputPath));
			Job job = Job.getInstance(jobConf);
			job.setMapperClass(ClassifierMapper.class);
			job.setCombinerClass(ClusterCombiner.class);
			job.setReducerClass(CenterReducer.class);
			FileInputFormat.addInputPath(job, new Path(dataPath));
			FileOutputFormat.setOutputPath(job, new Path(outputPath));
			if (job.waitForCompletion(true)) { // 开始执行并等待结束返回true
				if (convergence()) { // 判断是否收敛
					break;
				}
			}
		}
		// 收敛后，根据center进行聚类（使用map-reduce）
		// 目前是直接保存点的坐标
		// 改进方法：可以给每条数据增加编号，这样就能进行Combiner和Reducer来提高效率和节省空间
		Job job = Job.getInstance(jobConf);
		job.setMapperClass(ClassifierMapper.class);
		job.setReducerClass(PointCenterReducer.class);
		FileInputFormat.addInputPath(job, new Path(dataPath));
		FileOutputFormat.setOutputPath(job, new Path(outputPath));
		deletePath(fs, new Path(outputPath));
		job.waitForCompletion(true);
	}
	
	// 获得点坐标集
	public static ArrayList<Point> getPoints(String path) throws IOException {
		FileSystem fs = FileSystem.get(URI.create(namePath), conf);
		FSDataInputStream is = fs.open(new Path(path));
		LineReader lineReader = new LineReader(is, conf);
		Text line = new Text();
		ArrayList<Point> points = new ArrayList<>();
		while (lineReader.readLine(line) > 0) {
			Point point = new Point();
			point.readText(line);
			points.add(point);
		}
		lineReader.close();
		return points;
	}
	
	// 随机选择中心
	public static void randomCenters() throws IOException{
		ArrayList<Point> points = getPoints(dataPath);
		int size = points.size();
		HashSet<Integer> set = new HashSet<>();
		Random r = new Random();
		while (set.size() < centerNum) // 利用set的唯一性取中心数个不同的随机数
			set.add(r.nextInt(size));
		FileSystem fs = FileSystem.get(URI.create(namePath), conf);
		Path file = new Path(outputPath + resultName);
		deletePath(fs, file);
		OutputStream os = fs.create(file);
		BufferedWriter br = new BufferedWriter(new OutputStreamWriter(os, "UTF-8"));
		for (int index : set) { // 将中心写入文件
			br.write(points.get(index).toText().toString());
			br.write('\n');
		}
		br.close();
	}
	
	// 判断是否收敛
	public static boolean convergence() throws IOException {
		ArrayList<Point> oldCenters = getPoints(centerPath);
		ArrayList<Point> newCenters = getPoints(outputPath + resultName);
		for (int i = 0; i < centerNum; i++) {
			double dist = oldCenters.get(i).getDistance(newCenters.get(i));
			if (dist > 1e-6)
				return false;
		}
		return true;
	}
	
	// 如果存在则删除
	public static void deletePath(FileSystem fs, Path path) throws IOException {
		if (fs.exists(path)) fs.delete(path, true);
	}
	
}
