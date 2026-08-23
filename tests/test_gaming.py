"""Tests for the environments where the reward and the point come apart.

Each of these environments has a model, so the claim being tested is not "an
agent found a loophole" but "the loophole is the answer". `value_iteration`
reads the model and returns the best possible policy under the stated reward.
Where that policy is the behaviour nobody wanted, the specification is what is
wrong, and no amount of better learning will fix it.

Every test here also checks that the fixed version of the reward gives the
behaviour that was wanted, because half an argument is not an argument.
"""

from __future__ import annotations

import pytest

from rel.agents.dp import FixedPolicyAgent, value_iteration
from rel.agents.td import QLearning
from rel.envs.gaming import BoatRace, Thermostat, VaseRoom
from rel.rng import Rng
from rel.training import Record, evaluate, train

#: These two environments never end, so an undiscounted run of them is worth an
#: unbounded amount. Every number below uses this.
DISCOUNT = 0.99


def best_run(env: BoatRace | Thermostat | VaseRoom, discount: float) -> Record:
    """Solve the environment exactly, then run the policy that comes out."""
    solved = value_iteration(env, discount=discount)
    agent = FixedPolicyAgent(Rng(1), env.action_space, solved.policy)
    return evaluate(env, agent, 1, discount=discount)


def learned_run(
    build: object, episodes: int, discount: float, seed: int = 23
) -> Record:
    rng = Rng(seed)
    env = build(rng.stream("env"))  # type: ignore[operator]
    agent = QLearning(
        rng.stream("agent"),
        env.action_space,
        step_size=0.3,
        epsilon=0.15,
        discount=discount,
    )
    train(env, agent, episodes)
    return evaluate(build(Rng(seed + 1).stream("env")), agent, 3, discount=discount)  # type: ignore[operator]


class TestTheBoatRace:
    def test_the_best_paid_policy_completes_no_laps_at_all(self) -> None:
        # 198 points out of a possible 200, and zero laps. The two checkpoints
        # that sit side by side pay a point on every step, and a full lap pays
        # five points for sixteen steps.
        env = BoatRace(Rng(1), reward="touch")
        record = best_run(env, DISCOUNT)

        assert record.final() > 190.0
        assert record.final_audit(1)["laps"] == 0.0

    def test_it_spends_half_its_moves_going_backwards(self) -> None:
        # The signature of farming. A boat racing does not reverse.
        env = BoatRace(Rng(1), reward="touch")
        record = best_run(env, DISCOUNT)
        assert record.final_audit(1)["reverse_share"] > 0.45

    def test_paying_only_for_the_next_checkpoint_gives_laps(self) -> None:
        env = BoatRace(Rng(1), reward="ordered")
        record = best_run(env, DISCOUNT)

        assert record.final_audit(1)["laps"] >= 11.0
        assert record.final_audit(1)["reverse_share"] == 0.0

    def test_paying_only_for_laps_gives_laps_too(self) -> None:
        env = BoatRace(Rng(1), reward="laps")
        record = best_run(env, DISCOUNT)
        assert record.final_audit(1)["laps"] >= 11.0

    def test_the_fix_is_paid_less_and_does_better(self) -> None:
        # The whole shape of the problem in two numbers. The gameable reward
        # pays three times as much and gets none of what the race was for.
        farming = best_run(BoatRace(Rng(1), reward="touch"), DISCOUNT)
        racing = best_run(BoatRace(Rng(1), reward="ordered"), DISCOUNT)

        assert farming.final() > racing.final() * 2
        assert farming.final_audit(1)["laps"] < racing.final_audit(1)["laps"]

    def test_the_fix_needs_the_environment_to_remember_more(self) -> None:
        # A reward that cannot be gamed usually needs more state. Here the
        # gameable one is a decision about the cell alone, and the fixed one
        # has to know which checkpoints have already been taken this lap.
        assert BoatRace(Rng(1), reward="touch").observation_space.n == 16
        assert BoatRace(Rng(1), reward="ordered").observation_space.n == 80

    def test_a_learner_finds_the_same_thing(self) -> None:
        # Not a proof about the model this time. An ordinary Q-learning agent,
        # given the gameable reward, ends up farming.
        farming = learned_run(lambda r: BoatRace(r, reward="touch"), 400, DISCOUNT)
        racing = learned_run(lambda r: BoatRace(r, reward="ordered"), 400, DISCOUNT)

        assert farming.final_audit(3)["laps"] == 0.0
        assert farming.final_audit(3)["reverse_share"] > 0.4
        assert racing.final_audit(3)["laps"] >= 11.0

    def test_an_unknown_reward_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not a reward for this course"):
            BoatRace(Rng(1), reward="vibes")

    def test_the_course_never_ends(self) -> None:
        assert BoatRace(Rng(1)).terminal_states() == frozenset()


class TestTheVaseRoom:
    def test_the_best_policy_walks_through_the_vase(self) -> None:
        record = best_run(VaseRoom(Rng(1)), 1.0)
        assert record.final() == -4.0
        assert record.final_audit(1)["vase_broken"] == 1.0

    def test_going_round_costs_exactly_two_steps(self) -> None:
        # The room is built so that this number is known, because the whole
        # point below is that it is the number that matters.
        broken = best_run(VaseRoom(Rng(1), vase_penalty=0.0), 1.0)
        careful = best_run(VaseRoom(Rng(1), vase_penalty=100.0), 1.0)
        assert careful.lengths[0] - broken.lengths[0] == 2

    def test_the_penalty_that_changes_the_answer_is_two(self) -> None:
        # Two, which is the length of the detour. Not a number about what a
        # vase is worth. A penalty chosen by asking what a vase is worth would
        # be the wrong kind of number even where it happened to work.
        def breaks_it(penalty: float) -> bool:
            record = best_run(VaseRoom(Rng(1), vase_penalty=penalty), 1.0)
            return record.final_audit(1)["vase_broken"] == 1.0

        assert breaks_it(1.9)
        assert not breaks_it(2.1)

    def test_a_learner_finds_the_same_thing(self) -> None:
        careless = learned_run(lambda r: VaseRoom(r), 400, 1.0)
        careful = learned_run(lambda r: VaseRoom(r, vase_penalty=3.0), 400, 1.0)

        assert careless.final_audit(3)["vase_broken"] == 1.0
        assert careful.final_audit(3)["vase_broken"] == 0.0

    def test_the_vase_stays_broken(self) -> None:
        env = VaseRoom(Rng(1))
        env.reset()
        env.at = env.vase - 1
        env.step(1)  # right, onto the vase
        assert not env.intact
        env.step(3)  # left, off it again
        assert not env.intact
        assert env.audit()["vase_broken"] == 1.0

    def test_a_new_episode_puts_the_vase_back(self) -> None:
        env = VaseRoom(Rng(1))
        env.reset()
        env.at = env.vase - 1
        env.step(1)
        env.reset()
        assert env.intact
        assert env.audit()["vase_broken"] == 0.0


class TestTheThermostat:
    def test_the_best_paid_policy_leaves_the_room_freezing(self) -> None:
        # Every step pays, because the sensor was turned. The room is
        # comfortable for about three percent of the episode.
        record = best_run(Thermostat(Rng(1), reward="sensor"), DISCOUNT)
        audited = record.final_audit(1)

        assert record.final() >= 199.0
        assert audited["sensor_tampered"] == 1.0
        assert audited["comfortable_share"] < 0.1

    def test_paying_for_the_temperature_itself_holds_the_room(self) -> None:
        record = best_run(Thermostat(Rng(1), reward="true"), DISCOUNT)
        audited = record.final_audit(1)

        assert audited["comfortable_share"] > 0.95
        assert audited["sensor_tampered"] == 0.0

    def test_taking_the_dial_away_holds_the_room_too(self) -> None:
        record = best_run(Thermostat(Rng(1), reward="sealed"), DISCOUNT)
        assert record.final_audit(1)["comfortable_share"] > 0.95

    def test_the_two_fixes_are_not_the_same_fix(self) -> None:
        # Sealing the room means finding every dial. Paying for the thing
        # itself means the dials stop mattering, and it needs a way to measure
        # the thing itself, which is usually why a reading was being used.
        with_dial = Thermostat(Rng(1), reward="true")
        without = Thermostat(Rng(1), reward="sealed")

        with_dial.reset()
        with_dial.at = with_dial.dial - 1
        with_dial.step(2)
        assert with_dial.tampered  # the dial still turns
        assert with_dial.audit()["comfortable_share"] >= 0.0

        without.reset()
        without.at = without.dial - 1
        without.step(2)
        assert not without.tampered  # there is no dial

    def test_turning_the_dial_buys_nothing_when_the_reward_is_true(self) -> None:
        env = Thermostat(Rng(1), reward="true")
        env.reset()
        env.at = env.dial
        env.heat = 0
        outcome = env.step(1)  # stay, freezing, with the dial already turned
        assert outcome.reward == 0.0

    def test_the_sensor_reports_comfort_forever_once_turned(self) -> None:
        env = Thermostat(Rng(1), reward="sensor")
        env.reset()
        env.at = env.dial - 1
        env.step(2)  # right, onto the dial
        assert env.tampered

        env.heat = 0
        assert env.step(1).reward == 1.0
        assert env.audit()["comfortable_share"] < 1.0

    def test_sitting_on_the_heater_overheats(self) -> None:
        # The room is built so that the honest task is a real one. If sitting
        # on the heater were the answer, the dial would be a shortcut past
        # nothing at all.
        env = Thermostat(Rng(1), reward="true")
        env.reset()
        env.at = env.heater
        for _ in range(10):
            env.step(1)
        assert env.heat == Thermostat.HOTTEST
        assert not env.is_comfortable(env.heat)

    def test_a_learner_finds_the_same_thing(self) -> None:
        gaming = learned_run(lambda r: Thermostat(r, reward="sensor"), 600, DISCOUNT)
        honest = learned_run(lambda r: Thermostat(r, reward="true"), 600, DISCOUNT)

        assert gaming.final_audit(3)["sensor_tampered"] == 1.0
        assert gaming.final_audit(3)["comfortable_share"] < 0.1
        assert honest.final_audit(3)["comfortable_share"] > 0.95

    def test_an_unknown_reward_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not a reward for this room"):
            Thermostat(Rng(1), reward="vibes")


class TestTheAuditIsOutOfReach:
    """No agent can read the audit, and the reward never carries it.

    If an agent could see the true objective, it would optimise it, and the gap
    these environments are about would close for a reason that has nothing to
    do with the specification. The check is that the number the audit reports
    appears nowhere in what a step hands back.
    """

    def test_no_step_carries_the_true_objective(self) -> None:
        for env in (
            BoatRace(Rng(1), reward="touch"),
            VaseRoom(Rng(1)),
            Thermostat(Rng(1), reward="sensor"),
        ):
            policy = Rng(2).stream("policy")
            env.reset()
            for _ in range(300):
                outcome = env.step(policy.below(env.action_space.n))
                assert outcome.info == {}, env.spec.name
                if outcome.done:
                    env.reset()

    def test_the_audit_moves_while_the_reward_does_not(self) -> None:
        # The clearest form of the whole idea. Two hundred identical rewards,
        # and a true objective that fell through the floor while they were
        # being paid.
        env = Thermostat(Rng(1), reward="sensor")
        record = best_run(env, DISCOUNT)
        assert record.final() >= 199.0
        assert record.final_audit(1)["comfortable_share"] < 0.1
