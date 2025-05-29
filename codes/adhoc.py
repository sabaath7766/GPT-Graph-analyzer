import cv2
from pyzbar import pyzbar
import os

def find_qr_codes_in_video(video_path):
    """
    Scans a video file frame by frame to find and decode QR codes.

    Args:
        video_path (str): The path to the video file.

    Returns:
        list: A list of dictionaries, where each dictionary contains
              the QR code data and the timestamp (in seconds) where it was found.
              Returns an empty list if no QR codes are found or if the video
              cannot be opened.
    """
    # Check if the video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return []

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return []

    found_qr_codes = []
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS) # Get frames per second of the video

    if fps == 0: # Handle cases where fps might not be available or is zero
        print("Warning: Could not determine FPS of the video. Timestamps might be inaccurate or unavailable.")
        fps = 1 # Default to 1 to avoid division by zero, though timestamps will be just frame numbers

    print(f"Processing video: {video_path}")
    print(f"Video FPS: {fps if fps > 1 else 'N/A (timestamps will be frame numbers)'}")

    while cap.isOpened():
        # Read a frame from the video
        ret, frame = cap.read()

        # If frame is read correctly, ret is True
        if not ret:
            break # End of video

        frame_count += 1

        # Convert the frame to grayscale (pyzbar works better with grayscale images)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find QR codes in the current frame
        # The decode function returns a list of ZBarSymbol objects
        decoded_objects = pyzbar.decode(gray_frame)

        if decoded_objects:
            for obj in decoded_objects:
                # Extract the QR code data (it's in bytes, so decode to string)
                qr_data = obj.data.decode('utf-8')
                qr_type = obj.type

                # Calculate timestamp in seconds
                timestamp_seconds = frame_count / fps if fps > 0 else frame_count

                # Print information about the found QR code
                print(f"\n--- QR Code Found at Frame {frame_count} (approx. {timestamp_seconds:.2f}s) ---")
                print(f"  Type: {qr_type}")
                print(f"  Data: {qr_data}")

                # Store the found QR code data and timestamp
                found_qr_codes.append({
                    "data": qr_data,
                    "type": qr_type,
                    "frame": frame_count,
                    "timestamp_seconds": timestamp_seconds
                })

                # Optional: Display the frame with the QR code highlighted
                # (Uncomment the following lines if you want to see the video frames)
                # (You'll need to have a display environment for this to work)
                # points = obj.polygon
                # if len(points) > 4:
                #     hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
                #     cv2.polylines(frame, [hull], True, (0, 255, 0), 3)
                # else:
                #     cv2.polylines(frame, [np.array(points, dtype=np.int32)], True, (0, 255, 0), 3)
                #
                # cv2.imshow("QR Code Detected", frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'): # Press 'q' to quit preview
                #     cap.release()
                #     cv2.destroyAllWindows()
                #     return found_qr_codes
        else:
            # Print progress every 100 frames or so, to show it's working
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames...")


    # Release the video capture object
    cap.release()
    # cv2.destroyAllWindows() # Uncomment if you used cv2.imshow

    if not found_qr_codes:
        print("\nNo QR codes found in the video.")
    else:
        print(f"\nFinished processing. Found {len(found_qr_codes)} QR code(s).")

    return found_qr_codes

if __name__ == "__main__":
    # --- IMPORTANT: Replace this with the actual path to your video file ---
    video_file_path = "/Users/sabaath/Downloads/zackVideo.mp4"  # e.g., "/path/to/your/video.mp4" or "C:\\Users\\YourName\\Videos\\my_video.mov"

    # Check if the placeholder path is still there
    if video_file_path == "your_video.mp4":
        print("Please update the 'video_file_path' variable in the script with the actual path to your video file.")
    else:
        results = find_qr_codes_in_video(video_file_path)

        if results:
            print("\nSummary of QR Codes Found:")
            for i, result in enumerate(results):
                print(f"  {i+1}. Data: {result['data']}, Timestamp: {result['timestamp_seconds']:.2f}s (Frame: {result['frame']})")
