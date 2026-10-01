"""终端交互：解析命令、展示结果和错误。"""
import json
import sqlite3

import httpx
from openai import AsyncOpenAI, APITimeoutError, APIConnectionError, APIStatusError

from database import load_run, load_events
from runtime import execute_research, resume_research, get_resumable_runs


def show_recovery_hint() -> None:
    try:
        runs = get_resumable_runs()
    except sqlite3.Error:
        print("暂时无法读取恢复列表，可稍后输入 /resume 重试。")
        return
    if runs:
        title = " ".join(runs[0]["question"].split())[:100]
        print(f"发现 {len(runs)} 个可恢复任务，最近一个：{title}")
        print("输入 /continue 继续最近任务，或 /resume 按标题选择。")


def choose_run() -> str | None:
    runs = get_resumable_runs()
    if not runs:
        print("没有可恢复任务（需要有效的规划阶段或研究检查点）。")
        return None
    for index, run in enumerate(runs, start=1):
        title = " ".join(run["question"].split())[:100]
        status = "失败" if run["status"] == "failed" else "未完成"
        phase = {"planning": "待生成计划", "planned": "计划已保存", "research": "研究中"}[run["phase"]]
        print(f"{index}. {title} [{status}，{phase}，已保存 {run['evidence_count']} 条证据]")
    while True:
        selection = input("选择任务序号，回车或 0 取消：").strip()
        if selection in {"", "0"}:
            return None
        try:
            index = int(selection) - 1
        except ValueError:
            print("请输入列表中的数字序号。")
            continue
        if 0 <= index < len(runs):
            return runs[index]["id"]
        print("序号不在列表范围内。")


async def run_cli(client: AsyncOpenAI) -> None:
    print("研究 Agent 已启动。")
    print("输入问题开始研究；/resume 选择任务；/continue 继续最近任务；/exit 退出。")
    print("也支持 /show 任务编号、/events 任务编号、/resume 任务编号。")
    show_recovery_hint()

    while True:
        question = input("\n你：").strip()

        if question == "/exit":
            break

        if not question:
            print("问题不能为空。")
            continue
        command_parts = question.split(maxsplit=1)

        if command_parts[0] in {"/show", "/events"}:
            command = command_parts[0]
            if len(command_parts) != 2:
                print(f"用法：{command} 任务编号")
                continue

            run_id = command_parts[1].strip()

            try:
                saved_run = load_run(run_id)
                if command == "/events":
                    events = load_events(run_id)

            except LookupError:
                print("没有找到这个任务。")
                continue

            except (ValueError, TypeError):
                print("任务编号或记录格式不正确，无法读取。")
                continue

            except sqlite3.Error as error:
                print(f"数据库读取失败：{error}")
                continue

            print(f"\n[任务编号] {saved_run.id}")
            print(f"[研究问题] {saved_run.question}")
            print(f"[任务状态] {saved_run.status}")

            if command == "/events":
                print(f"[事件数量] {len(events)}")
                if not events:
                    print("这个任务还没有保存事件。")

                for index, event in enumerate(events, start=1):
                    print(
                        f"\n[{index}] {event['created_at']} "
                        f"{event['event_type']}"
                    )
                    print(json.dumps(event["payload"], ensure_ascii=False, indent=2))

                # 查看历史后返回输入提示，不创建或执行新的研究任务。
                continue

            print(f"[证据数量] {len(saved_run.evidence)}")

            if saved_run.answer is not None:
                print(f"\n[已保存的回答]\n{saved_run.answer}")

            if saved_run.error is not None:
                print(f"\n[失败原因] {saved_run.error}")

            continue

        is_resume = command_parts[0] == "/resume"
        is_continue = command_parts[0] == "/continue"
        if is_continue and len(command_parts) != 1:
            print("用法：/continue（不需要任务编号）")
            continue

        if question.startswith("/") and not (is_resume or is_continue):
            print("未知命令。支持 /resume、/continue、/show 任务编号、/events 任务编号、/exit。")
            continue

        try:
            if is_resume:
                run_id = command_parts[1].strip() if len(command_parts) == 2 else choose_run()
                if run_id is None:
                    continue
                run = await resume_research(client, run_id)
            elif is_continue:
                runs = get_resumable_runs()
                if not runs:
                    print("没有可恢复任务。")
                    continue
                print(f"继续研究：{' '.join(runs[0]['question'].split())[:100]}")
                run = await resume_research(client, runs[0]["id"])
            else:
                run = await execute_research(client, question)
        except APITimeoutError:
            print("模型请求超时，本次研究中止。")
            show_recovery_hint()
            continue

        except APIConnectionError:
            print("模型连接失败，本次研究中止。")
            show_recovery_hint()
            continue

        except APIStatusError as error:
            print(f"模型服务错误：HTTP {error.status_code}")
            if error.status_code in {401, 403}:
                print("请先检查模型密钥和访问权限，修正后再继续任务。")
            elif error.status_code == 402:
                print("请先检查模型服务余额，处理后再继续任务。")
            elif error.status_code == 429:
                print("请检查服务限流或额度；等待恢复或处理额度后再继续。")
            show_recovery_hint()
            continue

        except httpx.TimeoutException:
            print("搜索超时，本次研究中止。")
            show_recovery_hint()
            continue

        except httpx.RequestError:
            print("搜索连接失败，本次研究中止。")
            show_recovery_hint()
            continue

        except httpx.HTTPStatusError as error:
            print(
                "搜索服务错误："
                f"HTTP {error.response.status_code}"
            )
            show_recovery_hint()
            continue

        except ValueError as error:
            print(f"本次研究中止：{error}")
            show_recovery_hint()
            continue
        except LookupError as error:
            print(f"无法恢复：{error}")
            continue
        except sqlite3.Error as error:
            print(f"数据库操作失败，本次执行中止：{error}")
            continue

        print(f"\n助手：{run.answer}")
