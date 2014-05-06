
public class TimerMillis {
	
	long prevTime;
	long lastGap;
	
	public TimerMillis() {
		prevTime = System.currentTimeMillis();
	}
	

	public String getTime(String task) {
		long oldTime = prevTime;
		prevTime = System.currentTimeMillis();
		lastGap = prevTime - oldTime;
		return "Time for ["+task+"]: "+lastGap+"ms";
	}
	
	public void printTime(String task) {
		long oldTime = prevTime;
		prevTime = System.currentTimeMillis();
		lastGap = prevTime - oldTime;
		System.out.println("Time for ["+task+"]: "+lastGap+"ms");
	}
	
	public long lastGap() {
		return lastGap;
	}
}
