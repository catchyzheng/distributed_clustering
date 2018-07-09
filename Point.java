package clustering;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;

public class Point implements Writable {
	public ArrayList<Double> values;
	public static int length;
	
	public Point() {
		clear();
	}
	
	public Point(double value) {
		clear();
		for (int i = 0; i < length; i++)
			values.add(value);
	}
	
	public Point(Text text) {
		readText(text);
	}
	
	public void write(DataOutput out) throws IOException {
		for (int i = 0; i < length; i++)
			out.writeDouble(values.get(i));
	}
	
	public void clear() {
		if (values == null)
			values = new ArrayList<>();
		else values.clear();	
	}
	
	public void readFields(DataInput in) throws IOException {
		clear();
		for (int i = 0; i < length; i++)
			values.add(in.readDouble());
	}
	
	public void setValues(ArrayList<Double> values) {
		this.values = values;
	}
	
	public void readText(Text text) {
		clear();
		String[] fileds = text.toString().split(" ");
		for (int i = 0; i < length; i++) {
			values.add(Double.parseDouble(fileds[i]));
		}
	}
	
	public ArrayList<Double> getValues() {
		return values;
	}
	
	public double getDistance(Point p) {
		double dist = 0;
		for (int i = 0; i < length; i++) {
			dist += Math.pow(p.values.get(i)-values.get(i), 2);
		}
		return dist;
	}
	
	public void plus(Point p) {
		for (int i = 0; i < length; i++)
			values.set(i, values.get(i)+p.values.get(i));
	}
	
	public void mul(double value) {
		for (int i = 0; i < length; i++)
			values.set(i, values.get(i)*value);
	}
	
	public Text toText() {
		StringBuilder builder = new StringBuilder();
		for (int i = 0; i < length; i++)
			builder.append(String.format("%.4f", values.get(i))).append(' ');
		return new Text(builder.toString());
	}
	
}
