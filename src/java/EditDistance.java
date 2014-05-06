import java.util.Arrays;
import java.util.Scanner;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;


public class EditDistance {

	public static void main(String[] args) {

		int numThreads = 6;
		int sizeLeft = 50000, sizeRight = 50000;
		int rounds = 100 * 1000 * 1000 / (sizeLeft * sizeRight);
		rounds = 1;
		rounds /= 1;
		TimerMillis timer = new TimerMillis();
		String[] inputs = new String[rounds * 2];
		for (int i = 0; i < rounds * 2; i++) {
			inputs[i] = randomString(i % 2 == 0 ? sizeLeft : sizeRight);
		}

		timer.printTime("Generating " + rounds + " strings of size " + sizeLeft
				+ " and " + sizeRight);

		int[] distancesRow = new int[rounds], distancesDiag = new int[rounds], distancesDiagParallel = new int[rounds];
		//		for (int i = 0; i < rounds; i++) {
		//			editDistance(inputs[2 * i], inputs[2 * i + 1]);
		//		}
		//		timer.printTime("Edit Distance");

		for (int i = 0; i < rounds; i++) {
			distancesRow[i] = editDistanceRow(inputs[2 * i], inputs[2 * i + 1]);
		}
		timer.printTime("Edit Distance Row");

		for (int i = 0; i < rounds; i++) {
			distancesDiag[i] = editDistanceDiag(inputs[2 * i], inputs[2 * i + 1]);
		}
		timer.printTime("Edit Distance Diag");
		long diagGap = timer.lastGap();

		for (int i = 0; i < rounds; i++) {
			distancesDiagParallel[i] = editDistanceDiagParallel(inputs[2 * i], inputs[2 * i + 1], numThreads);
		}
		timer.printTime("Edit Distance Diag Parallel");
		long diagPGap = timer.lastGap();
		
		for (int i = 0; i < rounds; i++) {
			int edr = distancesRow[i];
			int edd = distancesDiag[i];
			int eddp = distancesDiagParallel[i];
			if (edr != edd || edr != eddp) {
				System.out.println("**ERR on " + i + ": edr: " + edr + " edd: "
						+ edd + " eddp: " + eddp + "**");
				System.out.println("S1: " + inputs[2 * i] + " S2: "
						+ inputs[2 * i + 1]);
			}
		}
		timer.printTime("Correctness Check");
		System.out.println("Speed up on "+numThreads+" threads: "+(double)diagGap/diagPGap);

		// Scanner scan = new Scanner(System.in);
		// while (true) {
		// String s1 = scan.nextLine(), s2 = scan.nextLine();
		// System.out.println(editDistance(s1,s2));
		// }
	}

	public static int editDistance(String s1, String s2) {
		// Make s2 longer
		if (s1.length() > s2.length()) {
			String temp = s1;
			s1 = s2;
			s2 = temp;
		}
		int n1 = s1.length(), n2 = s2.length();
		int[][] dp = new int[n1 + 1][n2 + 1];

		for (int i = 0; i < n1 + 1; i++)
			dp[i][0] = i;
		for (int i = 0; i < n2 + 1; i++)
			dp[0][i] = i;

		for (int r = 1; r < n1 + 1; r++) {
			for (int c = 1; c < n2 + 1; c++) {
				if (s1.charAt(r - 1) == s2.charAt(c - 1))
					dp[r][c] = dp[r - 1][c - 1];
				else
					dp[r][c] = 1 + Math.min(dp[r - 1][c],
							Math.min(dp[r - 1][c - 1], dp[r][c - 1]));
			}
		}
		return dp[n1][n2];
	}

	public static int editDistanceRow(String s1, String s2) {
		// Make s2 shorter
		if (s1.length() < s2.length()) {
			String temp = s1;
			s1 = s2;
			s2 = temp;
		}
		int n1 = s1.length(), n2 = s2.length();
		int[] prevRow = new int[n2 + 1];
		int[] curRow = new int[n2 + 1];

		for (int i = 0; i < n2 + 1; i++)
			prevRow[i] = i;

		for (int r = 1; r < n1 + 1; r++) {
			curRow[0] = r;
			for (int c = 1; c < n2 + 1; c++) {
				if (s1.charAt(r - 1) == s2.charAt(c - 1))
					curRow[c] = prevRow[c - 1];
				else
					curRow[c] = 1 + Math.min(prevRow[c],
							Math.min(prevRow[c - 1], curRow[c - 1]));
			}
			int[] temp = prevRow;
			prevRow = curRow;
			curRow = temp;
		}
		return prevRow[n2];
	}

	public static int editDistanceDiag(String s1, String s2) {
		// Make s2 longer
		if (s1.length() > s2.length()) {
			String temp = s1;
			s1 = s2;
			s2 = temp;
		}
		int n1 = s1.length(), n2 = s2.length();
		// int rounds = n1+n2+1;
		int[] prevDiagA = new int[n1 + 1], prevDiagB = new int[n1 + 1], curDiag = new int[n1 + 1];

		prevDiagA[0] = 0; // illustration
		prevDiagB[0] = 1;
		prevDiagB[1] = 1;

		for (int a = 2; a <= n1; a++) {
			curDiag[0] = a;
			curDiag[a] = a;
			for (int i = 1; i < a; i++) {
				int idx1 = a - i - 1, idx2 = i - 1;
				if (s1.charAt(idx1) == s2.charAt(idx2))
					curDiag[i] = prevDiagA[i - 1];
				else {
					int up = prevDiagB[i], left = prevDiagB[i - 1];
					int upLeft = prevDiagA[i - 1];
					curDiag[i] = 1 + Math.min(up, Math.min(left, upLeft));
				}
			}
			int[] tempA = prevDiagA;
			prevDiagA = prevDiagB;
			prevDiagB = curDiag;
			curDiag = tempA;
		}

		for (int b = 0; b < (n2 - n1); b++) {
			curDiag[n1] = n1 + b + 1;
			for (int i = 0; i < n1; i++) {
				int idx1 = n1 - i - 1, idx2 = b + i;
				if (s1.charAt(idx1) == s2.charAt(idx2))
					curDiag[i] = (b == 0 ? prevDiagA[i] : prevDiagA[i + 1]);
				else {
					int up = prevDiagB[i + 1], left = prevDiagB[i];
					int upLeft = (b == 0 ? prevDiagA[i] : prevDiagA[i + 1]);
					curDiag[i] = 1 + Math.min(up, Math.min(left, upLeft));
				}
			}
			int[] tempA = prevDiagA;
			prevDiagA = prevDiagB;
			prevDiagB = curDiag;
			curDiag = tempA;
		}

		for (int c = 0; c < n1; c++) {
			int diagSize = n1 - c;
			for (int i = 0; i < diagSize; i++) {
				int idx1 = n1 - i - 1, idx2 = (n2 - n1) + c + i;
				if (s1.charAt(idx1) == s2.charAt(idx2))
					curDiag[i] = ((n2 - n1 == 0) && (c == 0) ? prevDiagA[i]
							: prevDiagA[i + 1]);
				else {
					int up = prevDiagB[i + 1], left = prevDiagB[i];
					int upLeft = ((n2 - n1 == 0) && (c == 0) ? prevDiagA[i]
							: prevDiagA[i + 1]);
					curDiag[i] = 1 + Math.min(up, Math.min(left, upLeft));
				}
			}
			int[] tempA = prevDiagA;
			prevDiagA = prevDiagB;
			prevDiagB = curDiag;
			curDiag = tempA;
		}
		return prevDiagB[0];
	}

	public static int editDistanceDiagParallel(String s1s, String s2s,
			final int numThreads) {
		// Make s2 longer
		final String s1 = s1s.length() < s2s.length() ? s1s : s2s;
		final String s2 = s1s.length() < s2s.length() ? s2s : s1s;
		final int n1 = s1.length(), n2 = s2.length();

		// IMPORTANT
		final int serialRounds = Math.min(100, n1 + 1);
		
		final int[] finalPrevDiagA = new int[n1 + 1], finalPrevDiagB = new int[n1 + 1], finalCurDiag = new int[n1 + 1];

		final CyclicBarrier barrier = new CyclicBarrier(numThreads);

		Thread[] threads = new Thread[numThreads];
		for (int tid = 0; tid < numThreads; tid++) {
			threads[tid] = new Thread(new Runnable() {

				@Override
				public void run() {
					int tid = Integer
							.parseInt(Thread.currentThread().getName());
					int[] prevDiagA = finalPrevDiagA, prevDiagB = finalPrevDiagB, curDiag = finalCurDiag;
					if (tid == 0) {
						prevDiagA[0] = 0; // illustration
						prevDiagB[0] = 1;
						prevDiagB[1] = 1;

						for (int a = 2; a < serialRounds; a++) {
							curDiag[0] = a;
							curDiag[a] = a;
							for (int i = 1; i < a; i++) {
								int idx1 = a - i - 1, idx2 = i - 1;
								if (s1.charAt(idx1) == s2.charAt(idx2))
									curDiag[i] = prevDiagA[i - 1];
								else {
									int up = prevDiagB[i], left = prevDiagB[i - 1];
									int upLeft = prevDiagA[i - 1];
									curDiag[i] = 1 + Math.min(up,
											Math.min(left, upLeft));
								}
							}
							int[] tempA = prevDiagA;
							prevDiagA = prevDiagB;
							prevDiagB = curDiag;
							curDiag = tempA;
						}
						barrier();
					} else
						barrier();

					if (tid != 0) {
						switch (serialRounds % 3) {
						case 0: {
							int[] temp = curDiag;
							curDiag = prevDiagA;
							prevDiagA = prevDiagB;
							prevDiagB = temp;
						}; break;
						case 1: {
							int[] temp = curDiag;
							curDiag = prevDiagB;
							prevDiagB = prevDiagA;
							prevDiagA = temp;
						}; break;
						case 2: {
							// do nothing
						}; break;
						}
					}

					for (int a = serialRounds; a <= n1; a++) {
						if (tid == 0) {
							curDiag[0] = a;
							curDiag[a] = a;
						}
						int work = a - 1;
						int perWork = work / numThreads;

						for (int i = 1 + perWork * tid; i < ((tid == numThreads - 1) ? a
								: 1 + perWork * (tid + 1)); i++) {
							int idx1 = a - i - 1, idx2 = i - 1;
							if (s1.charAt(idx1) == s2.charAt(idx2))
								curDiag[i] = prevDiagA[i - 1];
							else {
								int up = prevDiagB[i], left = prevDiagB[i - 1];
								int upLeft = prevDiagA[i - 1];
								curDiag[i] = 1 + Math.min(up,
										Math.min(left, upLeft));
							}
						}
						int[] tempA = prevDiagA;
						prevDiagA = prevDiagB;
						prevDiagB = curDiag;
						curDiag = tempA;

						barrier();
					}


					for (int b = 0; b < (n2 - n1); b++) {
						if (tid == 0)
							curDiag[n1] = n1 + b + 1;
						int work = n1;
						int perWork = work/numThreads;

						for (int i = perWork * tid; i < ((tid == numThreads - 1) ? n1 : perWork * (tid+1)); i++) {
							int idx1 = n1 - i - 1, idx2 = b + i;
							if (s1.charAt(idx1) == s2.charAt(idx2))
								curDiag[i] = (b == 0 ? prevDiagA[i]
										: prevDiagA[i + 1]);
							else {
								int up = prevDiagB[i + 1], left = prevDiagB[i];
								int upLeft = (b == 0 ? prevDiagA[i]
										: prevDiagA[i + 1]);
								curDiag[i] = 1 + Math.min(up,
										Math.min(left, upLeft));
							}
						}
						int[] tempA = prevDiagA;
						prevDiagA = prevDiagB;
						prevDiagB = curDiag;
						curDiag = tempA;

						barrier();
					}

					for (int c = 0; c < n1 - serialRounds; c++) {
						int diagSize = n1 - c;
						int perWork = diagSize/numThreads;
						
						for (int i = perWork * tid; i < ((tid == numThreads - 1) ? diagSize : perWork * (tid+1)); i++) {
							int idx1 = n1 - i - 1, idx2 = (n2 - n1) + c + i;
							if (s1.charAt(idx1) == s2.charAt(idx2))
								curDiag[i] = ((n2 - n1 == 0) && (c == 0) ? prevDiagA[i]
										: prevDiagA[i + 1]);
							else {
								int up = prevDiagB[i + 1], left = prevDiagB[i];
								int upLeft = ((n2 - n1 == 0) && (c == 0) ? prevDiagA[i]
										: prevDiagA[i + 1]);
								curDiag[i] = 1 + Math.min(up,
										Math.min(left, upLeft));
							}
						}
						int[] tempA = prevDiagA;
						prevDiagA = prevDiagB;
						prevDiagB = curDiag;
						curDiag = tempA;
						
						barrier();
					}

					if (tid == 0) {
						for (int c = n1 - serialRounds; c < n1; c++) {
							int diagSize = n1 - c;
							
							for (int i = 0; i < diagSize; i++) {
								int idx1 = n1 - i - 1, idx2 = (n2 - n1) + c + i;
								if (s1.charAt(idx1) == s2.charAt(idx2))
									curDiag[i] = ((n2 - n1 == 0) && (c == 0) ? prevDiagA[i]
											: prevDiagA[i + 1]);
								else {
									int up = prevDiagB[i + 1], left = prevDiagB[i];
									int upLeft = ((n2 - n1 == 0) && (c == 0) ? prevDiagA[i]
											: prevDiagA[i + 1]);
									curDiag[i] = 1 + Math.min(up,
											Math.min(left, upLeft));
								}
							}
							int[] tempA = prevDiagA;
							prevDiagA = prevDiagB;
							prevDiagB = curDiag;
							curDiag = tempA;

						}
						
					}

				}

				private void barrier() {
					try {
						barrier.await();
					} catch (InterruptedException | BrokenBarrierException e) {
						e.printStackTrace();
					}
				}

			});
			threads[tid].setName(tid + "");
		}

		for (Thread t : threads)
			t.start();
		for (Thread t : threads)
			try {
				t.join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}

		switch ((n1 + n2 + 1) % 3) {
		case 0:
			return finalCurDiag[0];
		case 1:
			return finalPrevDiagA[0];
		case 2:
			return finalPrevDiagB[0];
		default:
			return -1;
		}

	}

	public static String randomString(int size) {
		StringBuilder builder = new StringBuilder();
		for (int i = 0; i < size; i++)
			builder.append((char) ((int) (Math.random() * 26) + 'a'));
		return builder.toString();
	}

}