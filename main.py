from fastapi import FastAPI, BackgroundTasks, Request, UploadFile, File, Form, Response, Depends, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
import uvicorn
import os
from video_pipeline import VideoPipeline, VideoConfig
import threading
from motor.motor_asyncio import AsyncIOMotorClient
import certifi
from dotenv import load_dotenv

# Load Env
load_dotenv("d:/JAK/.env")

# Create dirs if not exist (Must be before mounting)
os.makedirs("d:/JAK/static/css", exist_ok=True)
os.makedirs("d:/JAK/static/js", exist_ok=True)
os.makedirs("d:/JAK/templates", exist_ok=True)

app = FastAPI()

# Mounts
app.mount("/static", StaticFiles(directory="d:/JAK/static"), name="static")

# Templates
templates = Jinja2Templates(directory="d:/JAK/templates")

# DATABASE & AUTH CONFIG
# ==========================================
MONGO_URI = os.getenv("MONGO_URI")

# MongoDB Client
client = None
db = None
users_collection = None

@app.on_event("startup")
async def startup_db_client():
    global client, db, users_collection
    if MONGO_URI:
        try:
            # TLS CA File is often needed for Atlas on Windows
            client = AsyncIOMotorClient(MONGO_URI, tlsCAFile=certifi.where())
            db = client.get_database("ai_video_studio")
            users_collection = db.get_collection("users")
            
            # Verify connection
            await client.admin.command('ping')
            print(f"Connected to MongoDB Atlas: {db.name}")
        except Exception as e:
            print(f"MongoDB Connection Error: {e}")
            client = None
            users_collection = None
    else:
        print("WARNING: MONGO_URI not found in .env. Auth will fail.")

@app.on_event("shutdown")
async def shutdown_db_client():
    if client:
        client.close()


def get_password_hash(password):
    return password 

def verify_password(plain_password, stored_password):
    return plain_password == stored_password 

async def get_current_user(request: Request):
    username = request.cookies.get("session_token")
    if not username:
        return None
    if users_collection is not None:
        user = await users_collection.find_one({"username": username})
        return user
    return None


pipeline_instance = None
pipeline_thread = None

def selection_sort_users(users):

    n = len(users)
    for i in range(n):
        max_idx = i
        for j in range(i + 1, n):
            if users[j].get("video_count", 0) > users[max_idx].get("video_count", 0):
                max_idx = j
        users[i], users[max_idx] = users[max_idx], users[i]
    return users


@app.get("/leaderboard")
async def leaderboard(request: Request):
    user = await get_current_user(request)
    
    users_cursor = users_collection.find({})
    users_list = await users_cursor.to_list(length=1000)
    
    sorted_users = selection_sort_users(users_list)
    
    return templates.TemplateResponse("leaderboard.html", {
        "request": request,
        "user": user,
        "leaderboard": sorted_users
    })

@app.get("/")
async def home(request: Request):
    user = await get_current_user(request)
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "user": user
    })

@app.get("/register")
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/register")
async def register(request: Request, username: str = Form(...), password: str = Form(...)):
    if users_collection is None:
        return templates.TemplateResponse("register.html", {"request": request, "error": "Database not connected"})
    
    existing_user = await users_collection.find_one({"username": username})
    if existing_user:
        return templates.TemplateResponse("register.html", {"request": request, "error": "Username already taken"})
    
    new_user = {
        "username": username,
        "password": password, # Store plain text
        "video_count": 0
    }
    await users_collection.insert_one(new_user)
    return RedirectResponse(url="/login", status_code=303)

@app.get("/login")
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login(request: Request, response: Response, username: str = Form(...), password: str = Form(...)):
    if users_collection is None:
        return templates.TemplateResponse("login.html", {"request": request, "error": "Database not connected"})

    user = await users_collection.find_one({"username": username})
    if not user or password != user.get("password"):
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials"})
    
    # Create Session (Simple Plaintext)
    resp = RedirectResponse(url="/", status_code=303)
    resp.set_cookie(key="session_token", value=username, max_age=86400)
    return resp

@app.get("/logout")
async def logout(response: Response):
    resp = RedirectResponse(url="/login", status_code=303)
    resp.delete_cookie("session_token")
    return resp

@app.get("/api/status")
async def get_status():
    global pipeline_instance
    if pipeline_instance:
        return JSONResponse({
            "status": pipeline_instance.status,
            "progress": pipeline_instance.progress,
            "logs": pipeline_instance.logs[-10:] # Return last 10 logs
        })
    return JSONResponse({"status": "Idle", "progress": 0, "logs": []})

@app.post("/api/start")
async def start_generation(
    request: Request,
    background_tasks: BackgroundTasks,
    front: UploadFile = File(None),
    left: UploadFile = File(None),
    right: UploadFile = File(None),
    back: UploadFile = File(None)
):
    global pipeline_instance, pipeline_thread
    
    user = await get_current_user(request)
    if not user:
        return JSONResponse({"message": "Not authenticated"}, status_code=401)

    if pipeline_thread and pipeline_thread.is_alive():
        return JSONResponse({"message": "Already running"}, status_code=400)
    
    # Ensure upload dir exists
    upload_dir = "d:/JAK/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    # Save uploaded files
    saved_paths = {}
    for name, file_obj in [("front", front), ("left", left), ("right", right), ("back", back)]:
        if file_obj:
            file_location = os.path.join(upload_dir, f"{name}_{file_obj.filename}")
            with open(file_location, "wb+") as file_object:
                file_object.write(file_obj.file.read())
            saved_paths[name] = file_location
        else:
            saved_paths[name] = None

    pipeline_instance = VideoPipeline() # Re-init for fresh state
    
    # Update config with new paths if provided
    pipeline_instance.cfg.update_images(
        saved_paths.get("front"),
        saved_paths.get("left"),
        saved_paths.get("right"),
        saved_paths.get("back")
    )
    
    def run_job(username):
        pipeline_instance.run_full_pipeline()
        
        # Simple synchronous-style update for database (Easy Mode)
        if pipeline_instance.status == "Completed":
            import asyncio
            try:
                # We still need to run the async update, but let's keep it minimal
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(users_collection.update_one(
                    {"username": username},
                    {"$inc": {"video_count": 1}}
                ))
                loop.close()
                print(f"Video count saved for {username}")
            except Exception as e:
                print(f"Save error: {e}")

    pipeline_thread = threading.Thread(target=run_job, args=(user["username"],))
    pipeline_thread.start()
    
    return JSONResponse({"message": "Started", "status": "Initializing", "uploads": saved_paths})

@app.get("/video/{filename}")
async def get_video(filename: str):
    file_path = os.path.join("d:/JAK", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return JSONResponse({"error": "File not found"}, status_code=404)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
