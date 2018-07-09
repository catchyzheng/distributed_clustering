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
	private static int centerNum = 5; // ����
	private static int dim = 2; // ά��
	private static String namePath = "hdfs://192.168.15.128:9000";
	private static String dataPath = "/user/clustering/five_cluster";
	private static String outputPath = "/user/clustering/output";
	private static String centerPath = "/user/clustering/center";
	private static String resultName = "/part-r-00000";
	private static Configuration conf;
	// Mapper ����ÿһ����
	public static class ClassifierMapper extends Mapper<Object, Text, IntWritable, Point> {
		private ArrayList<Point> centers; // ����
		public void setup(Context context) 
				throws IOException, InterruptedException { // �������ļ��л�ȡ��ǰ������
			super.setup(context);
			centers = getPoints(centerPath);
		}
		// map ������ൽ�����������
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
	
	// Combiner ���ڽڵ����һ������
	public static class ClusterCombiner extends Reducer<IntWritable, Point, IntWritable, Point> {
		public void reduce(IntWritable key, Iterable<Point> points, Context context)
				throws IOException, InterruptedException {
			Point center = new Point(0);
			int cnt = 0;
			for (Point point : points) {
				center.plus(point);
				++cnt;
			}
			if (cnt > 0) // �������
				center.mul(1.0/cnt);
			context.write(key, center);
		}
	}
	
	// Reducer ��Combiner�õ������ĵ�ϲ�
	public static class CenterReducer extends Reducer<IntWritable, Point, Text, IntWritable> {
		public void reduce(IntWritable key, Iterable<Point> points, Context context)
				throws IOException, InterruptedException {
			Point center = new Point(0);
			int cnt = 0;
			for (Point point : points) {
				center.plus(point);
				++cnt;
			}
			if (cnt > 0) // �������
				center.mul(1.0/cnt);
			context.write(center.toText(), key);
		}
	}
	
	// Reducer ִֻ��һ�Σ���ÿ����ȷ�����ĵ㣨���ࣩ
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
		Point.length = dim; // ���õ��ά��
		randomCenters(); // ����������ĵ�output
		while (true) {
			// ����output��center
			FileUtil.copy(fs, new Path(outputPath + resultName), fs, new Path(centerPath), false, true, conf);
			// ɾ��output
			deletePath(fs, new Path(outputPath));
			Job job = Job.getInstance(jobConf);
			job.setMapperClass(ClassifierMapper.class);
			job.setCombinerClass(ClusterCombiner.class);
			job.setReducerClass(CenterReducer.class);
			FileInputFormat.addInputPath(job, new Path(dataPath));
			FileOutputFormat.setOutputPath(job, new Path(outputPath));
			if (job.waitForCompletion(true)) { // ��ʼִ�в��ȴ���������true
				if (convergence()) { // �ж��Ƿ�����
					break;
				}
			}
		}
		// �����󣬸���center���о��ࣨʹ��map-reduce��
		// Ŀǰ��ֱ�ӱ���������
		// �Ľ����������Ը�ÿ���������ӱ�ţ��������ܽ���Combiner��Reducer�����Ч�ʺͽ�ʡ�ռ�
		Job job = Job.getInstance(jobConf);
		job.setMapperClass(ClassifierMapper.class);
		job.setReducerClass(PointCenterReducer.class);
		FileInputFormat.addInputPath(job, new Path(dataPath));
		FileOutputFormat.setOutputPath(job, new Path(outputPath));
		deletePath(fs, new Path(outputPath));
		job.waitForCompletion(true);
	}
	
	// ��õ����꼯
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
	
	// ���ѡ������
	public static void randomCenters() throws IOException{
		ArrayList<Point> points = getPoints(dataPath);
		int size = points.size();
		HashSet<Integer> set = new HashSet<>();
		Random r = new Random();
		while (set.size() < centerNum) // ����set��Ψһ��ȡ����������ͬ�������
			set.add(r.nextInt(size));
		FileSystem fs = FileSystem.get(URI.create(namePath), conf);
		Path file = new Path(outputPath + resultName);
		deletePath(fs, file);
		OutputStream os = fs.create(file);
		BufferedWriter br = new BufferedWriter(new OutputStreamWriter(os, "UTF-8"));
		for (int index : set) { // ������д���ļ�
			br.write(points.get(index).toText().toString());
			br.write('\n');
		}
		br.close();
	}
	
	// �ж��Ƿ�����
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
	
	// ���������ɾ��
	public static void deletePath(FileSystem fs, Path path) throws IOException {
		if (fs.exists(path)) fs.delete(path, true);
	}
	
}
