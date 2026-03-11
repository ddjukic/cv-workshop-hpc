"""Generate synthetic PPE construction site images using Gemini 3 Pro."""

import mimetypes
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

PROMPTS = {
    # CCTV / security camera perspectives with mixed PPE compliance
    "cctv": [
        "Create a photorealistic CCTV security camera image of a construction site, timestamp overlay reading 08:15:22 AM - CAM 01 in top left corner, elevated angle looking down, 6 workers visible, only 2 wearing hard hats, others working without head protection, morning shadows, concrete foundation work",
        "Create a photorealistic security camera footage of a construction site at night, timestamp overlay 11:47:03 PM - CAM 03, floodlights illuminating the area, 4 workers doing night shift concrete pour, 2 wearing white hard hats 2 without any head protection, harsh artificial lighting",
        "Create a photorealistic CCTV camera view of construction site entrance gate, timestamp 06:52:18 AM - CAM 02, workers arriving for morning shift, some carrying helmets in hand not wearing them, others wearing helmets, turnstile visible, about half of 8 workers not wearing helmets",
        "Create a photorealistic overhead security camera image of a construction site, timestamp 14:23:45 PM - CAM 05, rainy weather, workers in rain jackets and hoods, 3 workers clearly without hard hats, 2 with hard hats over hoods, puddles on muddy ground, excavator in background",
        "Create a photorealistic CCTV security camera image from a crane-mounted camera looking down at construction workers, timestamp 10:31:07 AM - CAM 07, bird's eye view, 5 workers on scaffolding platform, 3 wearing orange hard hats 2 without, steel beam installation",
        "Create a photorealistic security camera footage of indoor construction area, timestamp 09:15:44 AM - CAM 04, building interior under renovation, poor fluorescent lighting, 5 workers doing drywall work, only 1 wearing a hard hat, others in baseball caps or nothing",
        "Create a photorealistic CCTV camera view of construction loading dock, timestamp 15:08:33 PM - CAM 06, trucks being unloaded, 6 workers visible, 2 forklift operators without helmets, 2 ground workers with helmets, 2 truck drivers without helmets, mixed scene",
        "Create a photorealistic security camera angle of a construction lunch break area next to active site, timestamp 12:15:22 PM - CAM 08, some workers eating with helmets removed on table, others still wearing helmets walking through, about half compliant",
        "Create a photorealistic CCTV overhead view of workers in a deep excavation trench, timestamp 07:44:19 AM - CAM 09, looking straight down, 4 workers in trench only 1 wearing hard hat, 2 workers at ground level both with helmets looking down into trench",
        "Create a photorealistic security camera footage of a construction site parking lot area, timestamp 16:42:11 PM - CAM 10, end of shift, workers walking to vehicles, most have removed their helmets and carry them, some left helmets behind, casual atmosphere",
    ],
    # Ground-level mixed compliance - explicitly non-compliant workers
    "mixed_compliance": [
        "Create a photorealistic construction site photo, 4 workers pouring concrete from a mixer truck, 2 wearing yellow hard hats and safety vests, 2 workers clearly without any head protection, one wiping forehead with arm, bright daylight",
        "Create a photorealistic close-up of 5 construction workers reviewing blueprints spread on a plywood table, only 1 wearing a hard hat, 2 holding helmets under their arm, 2 with no helmet visible at all, safety vests on all",
        "Create a photorealistic construction site scene with workers on scaffolding at different heights, workers on lower levels without hard hats, workers higher up wearing hard hats, clear height difference showing inconsistent compliance",
        "Create a photorealistic construction site with 2 visitors in business suits and no hard hats walking alongside 3 construction workers with helmets on, site office trailer in background, visitor badges visible",
        "Create a photorealistic construction site scene, one worker operating a jackhammer without any hard hat or ear protection, a supervisor nearby wearing a white hard hat pointing at something, 2 other workers in background one with helmet one without",
        "Create a photorealistic photo of a welder with welding mask flipped up and no hard hat underneath, sparks flying, a helper next to him holding materials wearing a hard hat, industrial construction setting, steel beams",
        "Create a photorealistic construction site rooftop scene, 5 roofers working, 3 without hard hats wearing only baseball caps backwards, 2 with proper hard hats, flat roof installation, city skyline in background",
        "Create a photorealistic construction trench scene, workers inside the trench without hard hats, their helmets sitting on the edge of the trench above them, 2 workers in trench without helmets, 1 supervisor at top with helmet",
        "Create a photorealistic construction site with electricians working on a partially built structure, 2 electricians on ladders without hard hats, 1 general laborer on ground with hard hat, wiring and conduit visible, interior framing",
        "Create a photorealistic demolition scene, 3 workers with sledgehammers breaking a wall, 2 without hard hats, 1 with hard hat, debris and dust flying, rubble on ground, clearly dangerous work without proper PPE",
    ],
    # Challenging conditions with mixed compliance
    "edge_cases": [
        "Create a photorealistic construction site in heavy rain, workers in rain gear with hoods pulled up, 2 workers clearly without hard hats under hoods, 2 with hard hats visible over rain gear, wet muddy ground, gray overcast sky",
        "Create a photorealistic construction site at high noon with bright sun causing lens flare and glare, workers with white hard hats hard to distinguish against bright sky, 3 of 6 workers clearly without hard hats, harsh shadows",
        "Create a photorealistic dusty demolition site with reduced visibility from dust clouds, some workers wearing hard hats barely visible through dust, 2 workers in foreground clearly without hard hats, rubble everywhere",
        "Create a photorealistic construction site at night with portable spotlights creating harsh shadows and bright spots, 6 workers visible at different distances, 3 without hard hats silhouetted against lights, dramatic lighting contrast",
        "Create a photorealistic construction site with a food truck and delivery person without PPE handing lunch to workers, some workers in line have removed helmets, others still wearing them, casual break-time atmosphere mixed with active work nearby",
        "Create a photorealistic wide-angle aerial drone photo of a massive construction site, tiny workers scattered across the frame, some wearing colorful helmets some clearly without, need to look closely to determine PPE compliance, scale makes detection challenging",
        "Create a photorealistic construction site with workers partially obscured behind parked trucks and heavy equipment, some hard hats visible poking above vehicle hoods, 2 workers clearly without helmets walking between vehicles",
        "Create a photorealistic construction site next to a glass building under construction, reflections in glass panels creating confusing mirror images of workers, 4 real workers visible with mixed helmet compliance, reflections adding visual noise",
        "Create a photorealistic construction site with dense rebar forest, workers threading through vertical rebar columns, helmets partially hidden by rebar creating occlusion, 3 workers with helmets 2 without, complex visual scene",
        "Create a photorealistic two-level construction site, upper floor workers wearing helmets looking down at ground floor workers without helmets, split-level view showing clear compliance difference between floors, stairs connecting levels",
    ],
    # Indoor warehouse and factory PPE scenes
    "warehouse": [
        "Create a photorealistic warehouse interior with high steel shelving racks, 5 workers operating forklifts and loading pallets, 2 forklift drivers wearing yellow hard hats, 3 ground workers without any head protection walking between aisles, overhead LED strip lighting, concrete floor with painted safety lines",
        "Create a photorealistic factory floor scene during a maintenance shutdown, 6 workers repairing conveyor belt machinery, 2 wearing white hard hats and face shields, 4 others crouching near belt rollers without helmets, industrial fluorescent lighting, oil stains on floor, toolboxes scattered around",
        "Create a photorealistic cold storage warehouse with visible breath vapor, 4 workers in heavy insulated jackets moving frozen goods on hand trucks, 1 wearing a hard hat over a beanie, 3 without any hard hats just wearing wool beanies, frost on metal shelving, dim yellowish lighting",
        "Create a photorealistic large distribution center with loading bays open to outside, 7 workers unloading delivery trucks, 3 wearing orange hard hats near the dock edge, 4 inside trucks stacking boxes without head protection, bright daylight streaming through open bay doors contrasting with dim interior",
        "Create a photorealistic warehouse mezzanine level scene, workers installing new racking systems, 3 workers on the mezzanine platform without hard hats leaning over railing, 2 workers below wearing hard hats looking up, steel beams and bolting equipment visible, high ceiling with skylights",
        "Create a photorealistic automotive parts warehouse, workers using overhead bridge crane to move heavy engine blocks, 1 crane operator in cab wearing hard hat, 4 workers on floor guiding the load with hand signals, only 1 ground worker wearing a hard hat, others bareheaded, grease-stained environment",
        "Create a photorealistic chemical storage warehouse with hazard placards on shelving, 5 workers doing inventory check with clipboards and tablets, 2 wearing hard hats and safety goggles, 3 without hard hats only wearing high-vis vests, spill containment pallets visible, harsh overhead sodium lighting",
        "Create a photorealistic textile factory interior with rows of industrial sewing machines, construction crew installing new ventilation ductwork overhead, 4 construction workers on scissor lifts, 2 with hard hats working on ducts, 2 without helmets holding ductwork sections, factory workers below in background",
        "Create a photorealistic food processing plant warehouse area under renovation, 6 workers installing stainless steel wall panels, 3 wearing white hard hats matching the sterile environment, 3 workers cutting panels on the ground without hard hats, bright clinical lighting, tiled floor partially exposed",
        "Create a photorealistic warehouse during night shift with emergency lighting active after a partial power outage, 5 workers navigating with flashlights, 2 wearing reflective hard hats that catch the light, 3 without hard hats barely visible in shadows, eerie green emergency exit signs glowing",
    ],
    # Road construction and highway work zone scenes
    "highway": [
        "Create a photorealistic highway construction zone at dawn, orange traffic cones and barrel barriers lining the work area, 6 workers in high-vis vests doing asphalt paving, 3 wearing hard hats near the paver machine, 3 without hard hats raking fresh asphalt, pink sunrise sky, headlights of passing cars in background",
        "Create a photorealistic road construction scene on a multi-lane highway, workers installing a concrete median barrier, 5 workers visible, 2 operating machinery with hard hats on, 3 hand-placing rebar without any head protection, traffic cones separating work zone from live lanes, midday sun overhead",
        "Create a photorealistic highway bridge deck repair scene, workers jackhammering deteriorated concrete on the bridge surface, 4 workers on the bridge deck, 2 with orange hard hats operating jackhammers, 2 without helmets shoveling debris, traffic passing on adjacent lanes, overcast gray sky",
        "Create a photorealistic night highway repaving operation with massive portable light towers illuminating the work zone, 7 workers in reflective gear, 3 wearing hard hats near the milling machine, 4 workers further back without head protection sweeping and directing traffic, orange flashing arrow board visible",
        "Create a photorealistic highway shoulder construction scene in summer heat, workers installing drainage culverts, 5 workers drenched in sweat, 2 wearing hard hats standing in the culvert trench, 3 above ground without hard hats drinking water from jugs, heat shimmer visible on asphalt, no clouds",
        "Create a photorealistic highway interchange construction site with elevated ramps being built, 8 workers on different levels, 4 workers on upper form-work wearing hard hats, 4 workers at ground level without helmets sorting lumber and rebar, cranes and heavy trucks in background, partly cloudy sky",
        "Create a photorealistic road construction scene during light rain, workers applying road line markings with a striping machine, 4 workers in rain jackets, 1 machine operator wearing hard hat, 3 walking behind checking line quality without hard hats hoods up, wet reflective road surface, mist in distance",
        "Create a photorealistic highway guardrail installation scene along a rural stretch, 5 workers using a pile driver to set posts, 2 workers near the pile driver wearing hard hats and ear protection, 3 workers carrying guardrail sections without any head protection, green hillside, scattered trees",
        "Create a photorealistic highway toll plaza construction zone, workers installing electronic toll gantry, 6 workers visible, 3 up on the gantry structure wearing hard hats and harnesses, 3 on the ground without hard hats handing up tools and cables, cars queuing in background, late afternoon golden light",
        "Create a photorealistic highway work zone with workers repairing a pothole cluster, 4 workers gathered around a large pothole, 1 operating a small roller with hard hat on, 3 shoveling and tamping hot asphalt without hard hats, traffic flaggers at each end of the zone, dust and steam rising from asphalt",
    ],
    # Easy, unambiguous images for error analysis — clear daylight, few workers,
    # obvious PPE compliance/non-compliance, no occlusion, simple backgrounds
    "easy": [
        "Create a photorealistic photo of 2 construction workers standing side by side facing the camera in bright daylight on a clean concrete slab, one wearing a bright yellow hard hat and orange safety vest, the other with no hat showing short brown hair, clear blue sky, simple building frame in background, sharp focus, no clutter",
        "Create a photorealistic photo of 3 construction workers standing upright in a row facing the camera on a sunny day, all three wearing bright orange hard hats and high-visibility vests, holding tools at their sides, plain gravel yard behind them, well-lit with no shadows on faces, clean simple scene",
        "Create a photorealistic photo of 2 workers on a clean construction site in full daylight, one worker wearing a white hard hat operating a wheelbarrow, the other worker with bare head no hat clearly visible standing next to a stack of lumber, blue sky, no obstructions between camera and workers",
        "Create a photorealistic photo of 3 construction workers facing the camera at a building entrance in bright morning light, 2 wearing red hard hats and 1 with no hard hat showing blonde hair, all wearing safety vests, clean paved ground, simple brick wall background, sharp clear image",
        "Create a photorealistic photo of 2 construction workers standing on a flat rooftop in clear sunny weather facing the camera, both without any hard hats or head covering, wearing just t-shirts and jeans with tool belts, clear sky behind them, simple flat roof surface, well-exposed bright image",
    ],
    # High-rise building construction and crane work scenes
    "highrise": [
        "Create a photorealistic high-rise construction site from ground level looking up, steel skeleton of a 20-story building, 5 workers visible on different floors, 3 on upper floors wearing hard hats walking on steel beams, 2 on a lower floor without hard hats sorting materials, tower crane overhead, blue sky",
        "Create a photorealistic high-rise concrete pour scene on an upper floor, workers guiding a concrete pump boom, 6 workers on the open floor slab, 3 wearing hard hats near the pump nozzle, 3 without helmets vibrating and screeding the fresh concrete, wind blowing their clothes, cityscape visible below",
        "Create a photorealistic high-rise construction site with workers installing glass curtain wall panels, 4 workers on a suspended scaffold on the building exterior, 2 wearing hard hats securing panels, 2 without hard hats handling suction cup lifters, dizzying height with city streets far below, partly cloudy",
        "Create a photorealistic tower crane operator cabin view looking down at the high-rise building top floor, 5 workers receiving a steel beam delivery from the crane hook, 2 wearing hard hats guiding the beam with tag lines, 3 without helmets waiting to bolt it in place, vertigo-inducing height perspective",
        "Create a photorealistic high-rise construction elevator lobby on the 15th floor, 7 workers crowding around the construction hoist, 3 exiting wearing hard hats, 4 waiting to go down have removed their helmets and hold them casually, exposed concrete columns and rebar stubs, dusty atmosphere",
        "Create a photorealistic high-rise rooftop scene with workers installing a mechanical penthouse, 5 workers with HVAC ducting and equipment, 2 wearing hard hats bolting down rooftop units, 3 without helmets running copper piping, panoramic city skyline in background, morning fog in valleys below",
        "Create a photorealistic high-rise construction site during windy conditions, tarps flapping on scaffolding, 4 workers on an exposed floor fighting the wind, 2 with hard hats strapped under chin holding onto railing, 2 without helmets crouching low for stability, dramatic cloudy sky, adjacent tall buildings",
        "Create a photorealistic high-rise construction site stairwell core, workers forming concrete walls for the elevator shaft, 6 workers in the confined vertical space, 3 wearing hard hats on scaffolding inside the shaft, 3 without helmets tying rebar at the base, dim work lights creating long shadows upward",
        "Create a photorealistic high-rise balcony installation scene, workers attaching precast balcony slabs to the building facade, 5 workers on various balcony levels, 3 with hard hats on the upper balconies using the mobile crane, 2 on a lower balcony without helmets applying sealant, sunset casting orange light on the building face",
        "Create a photorealistic high-rise construction site ground floor with the building core rising behind, 8 workers in the staging area, 4 wearing hard hats loading materials onto the hoist, 4 without hard hats organizing rebar bundles and form panels on the ground, tower crane silhouette against overcast sky, puddles from recent rain",
    ],
    # Close-up mixed-compliance scenes — 2-4 workers filling 20-40% of frame
    "close_up": [
        "Create a photorealistic close-up photo of 3 construction workers from the waist up at an outdoor concrete pour site, 2 wearing bright yellow hard hats and 1 bareheaded with short dark hair, all in orange safety vests, warm afternoon sunlight casting soft shadows, shallow depth of field, freshly poured concrete forms in background",
        "Create a photorealistic chest-level view of 2 warehouse workers filling the frame inside a steel storage facility, 1 wearing a white hard hat holding a clipboard, 1 without any head protection with a shaved head reaching for a shelf, fluorescent overhead lighting, metal racking blurred behind them, close framing",
        "Create a photorealistic close-up of 4 construction workers from the waist up on a rooftop work site, 1 wearing an orange hard hat pointing at ductwork, 3 without hard hats — one with curly brown hair holding a wrench, one with a buzz cut drinking from a water bottle, one with a grey beanie measuring tape in hand, overcast sky, HVAC units in background",
        "Create a photorealistic ground-level looking up photo of 2 construction workers at a tunnel entrance, filling the frame from waist up, 1 wearing a red hard hat with headlamp attachment holding a drill, 1 bareheaded with black hair wiping sweat from forehead, dim mixed lighting from tunnel interior and bright daylight outside, rock face texture visible",
        "Create a photorealistic close-up side profile view of 3 workers on a bridge deck construction site, 2 without hard hats — one with blonde ponytail carrying steel rebar bundles on shoulder, one with short red hair reading a blueprint — and 1 wearing a blue hard hat welding a guardrail bracket, golden hour sunlight from behind, river visible far below",
        "Create a photorealistic chest-level photo of 4 workers filling the frame at a power plant maintenance site, 2 wearing white hard hats with company logos inspecting a valve assembly, 1 bareheaded with grey hair holding a large pipe wrench, 1 without helmet wearing safety glasses pushing a tool cart, bright midday sun, industrial piping and catwalks in background",
        "Create a photorealistic close-up of 2 construction workers from the waist up inside a parking garage under construction, 1 wearing a yellow hard hat operating a concrete saw sending up dust, 1 without any hard hat with short brown hair holding a level against a pillar, harsh fluorescent work lights creating strong contrast, exposed concrete columns",
        "Create a photorealistic close-up photo of 3 workers filling the frame at a solar farm installation site, 1 wearing an orange hard hat tightening panel mounting brackets, 2 without hard hats — one with a dark complexion and close-cropped hair carrying a solar panel edge, one with a baseball cap backwards lifting wiring conduit, bright desert sun, rows of panels stretching into background",
        "Create a photorealistic slightly elevated view of 4 construction workers from the waist up at a water treatment plant, 3 wearing hard hats in different colors — 1 yellow 1 white 1 blue — inspecting a large pipe joint, 1 worker bareheaded with long dark hair tied back taking notes on a tablet, overcast diffused light, concrete settling tanks in background",
        "Create a photorealistic close-up of 2 shipyard workers filling the frame from the chest up next to a dry-docked vessel hull, 1 wearing a red hard hat with ear muffs around neck holding a grinding tool, 1 without any head protection with salt-and-pepper beard examining a weld seam, mixed indoor-outdoor lighting from open bay doors, massive steel hull plates behind them",
        "Create a photorealistic ground-level close-up of 3 construction workers from the waist up at an outdoor excavation site, 2 without hard hats — one with a shaved head pointing down into the trench, one with dark wavy hair using a phone to take a photo — and 1 wearing a bright orange hard hat shoveling dirt, early morning long shadows, backhoe arm visible behind them",
        "Create a photorealistic close-up photo of 4 workers filling the frame inside a warehouse renovation, 1 wearing a blue hard hat on a step ladder installing overhead cable tray, 2 without hard hats on the ground — one with short auburn hair feeding cable through conduit, one with a bandana holding wire cutters — and 1 wearing a yellow hard hat steadying the ladder, warm interior lighting mixing with daylight from open loading dock",
    ],
}


def generate_image(prompt: str, output_path: str) -> bool:
    """Generate a single image and save it."""
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        ),
    ]
    config = types.GenerateContentConfig(
        image_config=types.ImageConfig(
            aspect_ratio="16:9",
            image_size="1K",
        ),
        response_modalities=["IMAGE", "TEXT"],
    )

    try:
        for chunk in client.models.generate_content_stream(
            model="gemini-3-pro-image-preview",
            contents=contents,
            config=config,
        ):
            if chunk.parts is None:
                continue
            if chunk.parts[0].inline_data and chunk.parts[0].inline_data.data:
                inline_data = chunk.parts[0].inline_data
                file_extension = mimetypes.guess_extension(inline_data.mime_type)
                save_path = f"{output_path}{file_extension}"
                with open(save_path, "wb") as f:
                    f.write(inline_data.data)
                print(f"  Saved: {save_path}")
                return True
            else:
                if hasattr(chunk, "text") and chunk.text:
                    print(f"  Text response: {chunk.text[:100]}")
    except Exception as e:
        print(f"  ERROR: {e}")
        return False
    return False


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    synthetic_dir = os.path.join(base_dir, "synthetic_ppe")

    total = sum(len(v) for v in PROMPTS.values())
    generated = 0

    for category, prompts in PROMPTS.items():
        category_dir = os.path.join(synthetic_dir, category)
        os.makedirs(category_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Category: {category} ({len(prompts)} images)")
        print(f"{'='*60}")

        for i, prompt in enumerate(prompts):
            generated += 1
            print(f"\n[{generated}/{total}] Generating {category}_{i+1:02d}...")
            print(f"  Prompt: {prompt[:80]}...")

            output_path = os.path.join(category_dir, f"{category}_{i+1:02d}")

            # Skip if image already exists
            if any(
                os.path.exists(f"{output_path}{ext}")
                for ext in [".jpg", ".jpeg", ".png", ".webp"]
            ):
                print(f"  SKIPPED - already exists")
                continue

            success = generate_image(prompt, output_path)

            if not success:
                print(f"  FAILED - no image returned")

            # Rate limit: be gentle with the API
            time.sleep(3)

    print(f"\n{'='*60}")
    print(f"Done! Generated images in: {synthetic_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
