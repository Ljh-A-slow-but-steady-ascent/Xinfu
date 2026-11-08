from pathlib import Path

from backend.app.database import CaseRepository, init_database


FRONTEND = Path(__file__).resolve().parents[2] / "frontend"


def test_react_workspace_keeps_navigation_inside_single_page() -> None:
    source = (FRONTEND / "src" / "App.jsx").read_text(encoding="utf-8")
    package = (FRONTEND / "package.json").read_text(encoding="utf-8")

    assert '"react"' in package
    assert "xinfu_auth_token" in source
    assert "/v1/auth/me" in (FRONTEND / "src" / "api.js").read_text(encoding="utf-8")
    assert "function WorkspaceDrawer" in source
    assert "function Sidebar" in source
    assert "renameThread" in source
    assert "deleteThread" in source
    assert 'target="_top"' not in source


def test_case_summaries_never_expose_local_image_paths(tmp_path: Path) -> None:
    repository = CaseRepository(init_database("sqlite+pysqlite:///:memory:"))
    image = tmp_path / "lesion.png"
    image.write_bytes(b"image")
    repository.create_case("CASE_REACT", str(image), "hash", "local", "clinician")

    summaries = repository.list_cases()

    assert summaries[0]["case_id"] == "CASE_REACT"
    assert "local_image_path" not in summaries[0]


def test_workspace_folder_case_edit_and_placement(tmp_path: Path) -> None:
    repository = CaseRepository(init_database("sqlite+pysqlite:///:memory:"))
    user = repository.create_user("doctor", "hash", "医生")
    patient = repository.create_patient(
        name="原始姓名", sex="女", age=36, outpatient_no="OP-FOLDER",
        phone=None, medical_history="", allergy_history="", created_by=user.id,
    )
    image = tmp_path / "lesion.png"; image.write_bytes(b"image")
    repository.create_case("CASE_FOLDER", str(image), "hash", "local", "clinician")
    repository.link_case_patient(
        "CASE_FOLDER", patient.patient_id, "左前臂", "病灶变化",
        {"age": 36, "gender": "FEMALE", "region": "FOREARM", "itch": False,
         "hurt": False, "bleed": False, "grew": True, "changed": True,
         "diameter_1": 8, "diameter_2": 6, "skin_cancer_history": False,
         "cancer_history": False},
    )
    folder = repository.create_folder(user.id, "待复核病例")
    repository.move_workspace_item(user.id, "case", "CASE_FOLDER", folder["folder_id"])
    filtered = repository.list_cases(folder_id=folder["folder_id"], user_id=user.id)
    assert [item["case_id"] for item in filtered] == ["CASE_FOLDER"]

    updated = repository.update_case_record(
        "CASE_FOLDER", {"name": "已更新姓名", "lesion_site": "面部"}
    )
    assert updated["name"] == "已更新姓名"
    assert updated["lesion_site"] == "面部"


def test_react_intake_accumulates_previous_turns() -> None:
    source = (FRONTEND / "src" / "App.jsx").read_text(encoding="utf-8")
    markdown = (FRONTEND / "src" / "RichMarkdown.jsx").read_text(encoding="utf-8")
    assert "active.intakeDraft" in source
    assert "accumulatedNarrative" in source
    assert "active.preparedIntake" in source
    assert "resumesPreparedCase" in source
    assert "recoverIntakeNarrative(active.messages)" in source
    assert 'intakeState: "READY"' in source
    assert "请补充：" in source
    assert "/v1/agent/documents/upload" in source
    assert "/v1/workspace/folders" in source
    assert "function looksLikeClinicalIntakeDocument" in source
    assert "已读取病例信息，附病灶图片后即可开始分析" in source
    assert 'import("./RichMarkdown")' in source
    assert "function SecureMarkdownImage" in markdown
    assert "图片加载中" in markdown
    assert "fetchBlob(text, token)" in markdown
    assert "确认删除报告" not in source
    assert "请输入 ${id} 确认" in source
