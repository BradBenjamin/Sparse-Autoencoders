import psutil

def clear_port_safely(port: int):
    killed_any = False
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if any(conn.laddr.port == port for conn in proc.connections(kind='inet')):
                print(f"Killing {proc.info['name']} (PID: {proc.info['pid']}) on port {port}")
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except psutil.TimeoutExpired:
                    proc.kill()  # Force kill only if SIGTERM timed out
                killed_any = True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if not killed_any:
        print(f"Port {port} is already free.")
