package Run3;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Calendar;

public class Log {
	PrintWriter logger;
	Calendar cal;
	DateFormat df = new SimpleDateFormat("HH:mm:ss");
	Log(String filepath) {
		try {
			logger = new PrintWriter(filepath);
		} catch (FileNotFoundException e) {e.printStackTrace();}
	}
	
	public void log(String content) {
		cal = Calendar.getInstance();
		System.out.println(df.format(cal.getTime())+" - "+content);
		logger.println(df.format(cal.getTime())+" - "+content);
		logger.flush();
	}
	
	public void close() {
		logger.close();
	}
}
