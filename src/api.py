import sys
import os

# Ensure src directory is on the path so sibling modules can be imported
sys.path.insert(0, os.path.dirname(__file__))

import json
import asyncio
from dataclasses import asdict
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from data_fetcher import StockDataFetcher, StockData
import crew_agentic_workflow
from crewai import Crew, Task

app = FastAPI(title="AI Trading Debate API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ──────────────────────────────────────────────────

class StockDataRequest(BaseModel):
    symbol: str
    openai_key: str
    reddit_client_id: str = ""
    reddit_client_secret: str = ""


class DebateRequest(BaseModel):
    symbol: str
    openai_key: str
    max_rounds: int = 3
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    bull_risk_tolerance: str = "Medium-High"
    bull_focus_areas: list[str] = ["Breakouts", "Momentum"]
    bull_style: str = "Aggressive momentum trader seeking upside breakouts"
    bull_beliefs: str = "Markets trend and momentum persists; breakouts lead to sustained moves"
    bear_risk_tolerance: str = "Low-Medium"
    bear_focus_areas: list[str] = ["Risk Factors", "Overbought Conditions"]
    bear_style: str = "Defensive risk manager focused on capital preservation"
    bear_beliefs: str = "Markets mean-revert; protecting capital is more important than chasing gains"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _stock_data_to_dict(stock_data: StockData) -> dict:
    """Convert StockData dataclass to a JSON-serialisable dict."""
    d = asdict(stock_data)
    # asdict recurses into nested dataclasses; dicts/None values are fine as-is
    return d


def _sse(event: dict) -> str:
    """Format a dict as a Server-Sent Events data line."""
    return f"data: {json.dumps(event)}\n\n"


def _fetch_stock_data(req_symbol: str, reddit_client_id: str, reddit_client_secret: str) -> StockData:
    fetcher = StockDataFetcher(
        reddit_client_id=reddit_client_id or None,
        reddit_client_secret=reddit_client_secret or None,
    )
    return fetcher.get_stock_data(req_symbol)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/stock-data")
def get_stock_data(req: StockDataRequest):
    stock_data = _fetch_stock_data(req.symbol, req.reddit_client_id, req.reddit_client_secret)
    return _stock_data_to_dict(stock_data)


@app.post("/api/debate/stream")
def stream_debate(req: DebateRequest):
    """Stream the full debate as Server-Sent Events."""

    async def generate() -> AsyncGenerator[str, None]:
        try:
            # ── 1. Fetch stock data ────────────────────────────────────────
            yield _sse({"type": "status", "message": f"Fetching stock data for {req.symbol}..."})
            await asyncio.sleep(0)  # allow the event to flush

            loop = asyncio.get_event_loop()
            stock_data: StockData = await loop.run_in_executor(
                None,
                _fetch_stock_data,
                req.symbol,
                req.reddit_client_id,
                req.reddit_client_secret,
            )
            yield _sse({"type": "stock_data", "data": _stock_data_to_dict(stock_data)})
            await asyncio.sleep(0)

            # ── 2. Create CrewAI agents ────────────────────────────────────
            yield _sse({"type": "status", "message": "Initialising CrewAI agents..."})
            await asyncio.sleep(0)

            bull_focus = ", ".join(req.bull_focus_areas)
            bear_focus = ", ".join(req.bear_focus_areas)

            researcher, bull_agent, bear_agent, judge_agent = await loop.run_in_executor(
                None,
                lambda: crew_agentic_workflow.create_trading_agents(
                    bull_focus=bull_focus,
                    bear_focus=bear_focus,
                    api_key=req.openai_key,
                ),
            )

            # ── 3. Research phase ──────────────────────────────────────────
            yield _sse({"type": "status", "message": f"Researcher is gathering data for {req.symbol}..."})
            await asyncio.sleep(0)

            def _run_research():
                research_task = Task(
                    description=(
                        f"Use your tools to gather all available technical data, recent news, "
                        f"and Reddit sentiment for the stock {req.symbol}. "
                        f"Compile this into a comprehensive research report for the analysts."
                    ),
                    expected_output=(
                        f"A detailed dossier containing technical levels, recent news context, "
                        f"and social sentiment for {req.symbol}."
                    ),
                    agent=researcher,
                )
                result = Crew(agents=[researcher], tasks=[research_task], verbose=False).kickoff()
                return research_task.output.raw or str(result)

            research_res: str = await loop.run_in_executor(None, _run_research)
            yield _sse({"type": "research", "content": research_res})
            await asyncio.sleep(0)

            # ── 4. Debate rounds ───────────────────────────────────────────
            last_bull_argument = "No previous arguments."
            last_bear_argument = "No previous arguments."

            for round_num in range(1, req.max_rounds + 1):
                # Bull turn
                yield _sse({"type": "status", "message": f"Agent Bull is formulating case (Round {round_num})..."})
                await asyncio.sleep(0)

                bull_desc = f"""
                    Read this background research:
                    {research_res}

                    The Bear analyst recently argued: {last_bear_argument}

                    Construct a highly persuasive 2-3 paragraph argument/rebuttal on why we should
                    BUY or GO LONG on {req.symbol} today.
                    Directly attack weaknesses in the bear argument using the technical and news data.
                """

                def _run_bull(desc=bull_desc):
                    bull_task = Task(
                        description=desc,
                        expected_output="A 2-3 paragraph bullish trading thesis citing data.",
                        agent=bull_agent,
                    )
                    result = Crew(agents=[bull_agent], tasks=[bull_task], verbose=False).kickoff()
                    return bull_task.output.raw or str(result)

                bull_res: str = await loop.run_in_executor(None, _run_bull)
                last_bull_argument = bull_res
                yield _sse({"type": "bull", "round": round_num, "content": bull_res})
                await asyncio.sleep(0)

                # Bear turn
                yield _sse({"type": "status", "message": f"Agent Bear is formulating counter-argument (Round {round_num})..."})
                await asyncio.sleep(0)

                bear_desc = f"""
                    Read this background research:
                    {research_res}

                    The Bull analyst recently argued: {last_bull_argument}

                    Construct a highly persuasive 2-3 paragraph argument/rebuttal on why we should
                    SELL or GO SHORT on {req.symbol} today.
                    Directly attack weaknesses in the bull argument using the technical and news data.
                """

                def _run_bear(desc=bear_desc):
                    bear_task = Task(
                        description=desc,
                        expected_output="A 2-3 paragraph bearish trading thesis citing data.",
                        agent=bear_agent,
                    )
                    result = Crew(agents=[bear_agent], tasks=[bear_task], verbose=False).kickoff()
                    return bear_task.output.raw or str(result)

                bear_res: str = await loop.run_in_executor(None, _run_bear)
                last_bear_argument = bear_res
                yield _sse({"type": "bear", "round": round_num, "content": bear_res})
                await asyncio.sleep(0)

            # ── 5. Judge verdict ───────────────────────────────────────────
            yield _sse({"type": "status", "message": "Chief Risk Officer is rendering final verdict..."})
            await asyncio.sleep(0)

            judge_desc = f"""
                Review the full research report for {req.symbol}:
                {research_res}

                The Final Bull argument is: {last_bull_argument}
                The Final Bear argument is: {last_bear_argument}

                Deliver a final recommendation formatted strictly in markdown with:
                - **Decision**: BUY / SELL / HOLD
                - **Confidence**: 1-10
                - **Entry / Target / Stop Loss Levels**
                - **Reasoning**: A concise summary of why you chose this path over the alternatives.
            """

            def _run_judge():
                judge_task = Task(
                    description=judge_desc,
                    expected_output="A structured markdown trading recommendation with clear levels and reasoning.",
                    agent=judge_agent,
                )
                result = Crew(agents=[judge_agent], tasks=[judge_task], verbose=False).kickoff()
                return judge_task.output.raw or str(result)

            judge_res: str = await loop.run_in_executor(None, _run_judge)
            yield _sse({"type": "verdict", "content": judge_res})
            await asyncio.sleep(0)

            yield _sse({"type": "complete"})

        except Exception as exc:
            yield _sse({"type": "error", "message": str(exc)})

    return StreamingResponse(generate(), media_type="text/event-stream")
