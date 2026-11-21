using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace SoccerReplay.EditorTools
{
    public static class ReplaySceneSetup
    {
        const string ScenePath = "Assets/Scenes/Replay.unity";

        [MenuItem("Soccer Replay/Create Replay Scene")]
        public static void CreateScene()
        {
            if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;

            var scene = EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);

            var cam = Camera.main;
            if (cam != null)
            {
                cam.fieldOfView = 45f;
                cam.farClipPlane = 500f;
                cam.transform.position = new Vector3(0f, 24f, -62f);
                cam.transform.LookAt(Vector3.zero);
            }

            var light = Object.FindAnyObjectByType<Light>();
            if (light != null) light.transform.rotation = Quaternion.Euler(55f, -30f, 0f);

            var replay = new GameObject("Replay");
            replay.AddComponent<ReplayController>();
            Selection.activeGameObject = replay;

            EditorSceneManager.SaveScene(scene, ScenePath);
            Debug.Log($"[Replay] Created {ScenePath}. Press Play to watch the match.");
        }

        [MenuItem("Soccer Replay/Create Replay Scene and Play")]
        public static void CreateAndPlay()
        {
            CreateScene();
            EditorApplication.EnterPlaymode();
        }
    }
}
