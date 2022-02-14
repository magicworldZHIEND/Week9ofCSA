import java.math.BigInteger;
import java.util.*;

/**
 * @Classname : main //类名
 * @Description: CSA第九周数据结构练习, 题目来源为力扣第278场周赛 //描述
 * @Author : Administrator //作者
 * @Date : 2022/2/14 13:59//日期
 */
public class main {
    public static void main(String[] args) {
        System.out.println("----------第一题---------");
        int nums[] = {4, 2, 6, 1, 12};
        int original = 2;
        System.out.println(findFinalValue(nums, original));
        System.out.println("----------第二题---------");
        int nums2[] = { 1,1};
        System.out.println(maxScoreIndices(nums2));
        System.out.println("----------第三题---------");
        System.out.println(subStrHash("fbxzaad",31,100,3,32));
        System.out.println("----------第四题---------");
        System.out.println(Arrays.toString(groupStrings(new String[]{"a", "b", "ab", "cde"})));
    }

    /**
     * 1. 将找到的值乘以 2
     * 给你一个整数数组 nums ，另给你一个整数 original ，这是需要在 nums 中搜索的第一个数字。
     * 接下来，你需要按下述步骤操作：
     * 情况1.如果在 nums 中找到 original ，将 original乘以 2 ，得到新 original（即，令 original = 2 * original）。
     * 情况2.否则，停止这一过程。
     * 情况3.只要能在数组中找到新 original ，就对新 original 继续 重复 这一过程。
     * 返回 original 的最终值。
     */
    public static int findFinalValue(int[] nums, int original) {
        for (int i = 0; i < nums.length; ++i) {
            if (nums[i] == original) {
                // 找到后*2
                original = original * 2;
                // 重新开始遍历
                i = -1;
            }
        }
        return original;
    }

    /**
     * 2. 分组得分最高的所有下标
     * 给你一个下标从 0 开始的二进制数组 nums ，数组长度为 n 。nums 可以按下标 i（ 0 <= i <= n ）拆分成两个数组（可能为空）：numsleft 和 numsright 。
     * numsleft 包含 nums 中从下标 0 到 i - 1 的所有元素（包括 0 和 i - 1 ），而 numsright 包含 nums 中从下标 i 到 n - 1 的所有元素（包括 i 和 n - 1 ）。
     * 如果 i == 0 ，numsleft 为 空 ，而 numsright 将包含 nums 中的所有元素。
     * 如果 i == n ，numsleft 将包含 nums 中的所有元素，而 numsright 为 空 。
     * 下标 i 的 分组得分 为 numsleft 中 0 的个数和 numsright 中 1 的个数之 和 。
     * 返回 分组得分 最高 的 所有不同下标 。你可以按 任意顺序 返回答案。
     * 知识点:
     * 1. Int数组转为Interger数组
     * 思路:
     * 1. 计算数组中所有0和1的总数量
     * 2. 将左为空, 右为数组的情况设置为初始态
     * 3. 依次判断数组中的元素, 如果为0 , 左分数+1, 如果为1,右分数-1
     * 4. 每个位置的分数与已知最大数进行判断, 如果大于, 那么动态数组清空,重新加入 , 如果等于, 一并加入, 如果小于则无视
     */
    public static List<Integer> maxScoreIndices(int[] nums) {
        // 先将Int数组装维数值流
        //IntStream stream = Arrays.stream(nums);
        // 流中的元素全部装箱, 转换为流(int 转为 Integer)
        //Stream<Integer> integerStream = stream.boxed();
        // 流在转化为数组
        //Integer[] integers = integerStream.toArray(Integer[]::new);
        // 创建数组计算存储
        int numsleft[] = {}; // 左数组
        int numsright[] = {}; // 右数组
        List<Integer> maxIndex = new ArrayList<>(); // 得分最大的下标集合\
        int score = 0; // 当前分数
        int scoreLeft = 0;
        int scoreRight = 0;
        int maxScore = 0; // 最大分数
        // 算法时间过长:
        /* for(int i = 0;i<=nums.length;++i){
            if(i==0){
                numsright = nums;
            }else if(i == nums.length){
                numsleft =  nums;
            }else{
                numsleft = new int[i];
                System.arraycopy(nums,0,numsleft,0,i);
                numsright = new int[nums.length-i];
                System.arraycopy(nums,i,numsright,0,nums.length-i);
            }
            for(int j:numsleft){
                if(j==0){
                    scoreLeft++;
                }
            }
            for(int j:numsright){
                if(j==1){
                    scoreRight++;
                }
            }
            score = scoreLeft+scoreRight;
            //一旦发现当前的score > maxScore 就要对maxIndex进行清空重新添加
            if(score>maxScore){
                maxScore = score;
                maxIndex.clear();
                maxIndex.add(i);
            }else if (score == maxScore){
                maxIndex.add(i);
            }
        }*/
        int zeroSum = 0;
        int oneSum = 0;
        for (int i : nums) {
            if (i == 0) {
                zeroSum++;
            } else {
                oneSum++;
            }
        }
        // 当做左列表为空时
        scoreLeft = 0;
        scoreRight = oneSum;
        maxScore = scoreLeft+scoreRight;
        maxIndex.add(0);
        // 左列表不为空
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == 0) {
                scoreLeft++;
            }
            if (nums[i] == 1) {
                scoreRight--;
            }
            score = scoreLeft + scoreRight;
            if (score > maxScore) {
                maxScore = score;
                maxIndex.clear();
                maxIndex.add(i + 1);
            } else if (score == maxScore) {
                maxIndex.add(i + 1);
            }
        }
        return maxIndex;
    }

    /**
     * 3. 查找给定哈希值的子串
     * 给定整数 p和 m，一个长度为 k且下标从 0开始的字符串s的哈希值按照如下函数计算：
     * hash(s, p, m) = (val(s[0]) * p0 + val(s[1]) * p1 + ... + val(s[k-1]) * pk-1) mod m.
     * 其中val(s[i])表示s[i]在字母表中的下标，从val('a') = 1 到val('z') = 26。
     * 给你一个字符串s和整数power，modulo，k和hashValue。请你返回 s中 第一个 长度为 k的 子串sub，满足hash(sub, power, modulo) == hashValue。
     * 测试数据保证一定 存在至少一个这样的子串。
     * 子串 定义为一个字符串中连续非空字符组成的序列
     */
    public static String subStrHash(String s, int power, int modulo, int k, int hashValue) {
        //1. 参数规范化
        String string = s;
        int length = k;
        ArrayList<String> strings = new ArrayList<>();
        //2. 对字符串进行划分(滑动窗口原理, 每次前进一个字符)
        for(int i =0;i<string.length();i++){
            if(i+length<string.length()) {
                strings.add(string.substring(i, i + length));
            }
            else{
                strings.add(string.substring(i,string.length()));
            }
        }

        for (int i = 0; i < strings.size(); i++) {
            BigInteger stringHashValue = new BigInteger("0");
            //3. 先将字符串数组中所有字符串转变成数字
            int nums[] = new int[strings.get(i).length()];
            for(int j = 0 ; j<strings.get(i).length();j++)
            {
                int number = (int)strings.get(i).charAt(j)-96;
                nums[j] = number;
            }
            //4. 求出字符串数组中每个字符串的hash对应值
            for(int m =0 ; m<strings.get(i).length();m++){
                //为了防止数字溢出并且降低复杂度，我们需要用到下面的公式:
                //a mod c = (a mod c) mod c
                //(a+b)mod c = (a mod c + b mod c) mod c
                //(a * b) mod c= (a mod c * b mod c)mod c
                // 数据类型转换:
                BigInteger Nums = BigInteger.valueOf(nums[m]) ;
                BigInteger Power = BigInteger.valueOf(power);
                BigInteger M = BigInteger.valueOf(m);
                BigInteger Modulo = BigInteger.valueOf(modulo);
                stringHashValue = stringHashValue.add(Nums.mod(Modulo).multiply(Power.modPow(M,Modulo)));
            }
            if(stringHashValue.mod(BigInteger.valueOf(modulo)).equals(BigInteger.valueOf(hashValue))){
                return strings.get(i);
            }
        }
        return null;
    }

    public static String subStrHash1(String s, int power, int modulo, int k, int hashValue) {
        //1. 参数规范化
        String string = s;
        int length = k;
        // 计算第一个子串"LeetCode"  size=5  k=2  第一个子串[4,3]
        char[] arr = string.toCharArray(); // 将字符串变为字符数组
        int ans = 0, n = string.length();
        long hash = 0, mod = 1;
        // k-1个初始化
        for (int i = n - 1; i >= n - length + 1; i--) {
            mod = mod * power % modulo;
            hash = power * (hash + index(arr[i])) % modulo;
        }
        // 一次计算前面的子串
        for (int i = s.length() - k; i >= 0; i--) {
            // k-1个加上头的1个，凑k个  第一个是p^0=1 即charIndex本身
            hash = (index(arr[i]) + hash) % modulo;
            if (hash == hashValue) {
                ans = i;
            }
            // 去除最后一个，给下一个子串用  最后一个(charIndex*power^(k-1)) % p = (charIndex * (power^(k-1) % p)) % p
            // ∵ power^(k-1)%p = mod
            // ∴ (charIndex * (power^(k-1) % p)) % p = (charIndex*mod) %p
            long lastHash = (index(arr[i + k - 1]) * mod) % modulo;
            // 把去除最后一个字符的hash*power当做下一个子串的后k-1个字符的hash值
            // hash - lastHash 就行，但是防止hash更小出现负数，所以+modulo。
            hash = power * (hash - lastHash + modulo) % modulo;
        }
        return string.substring(ans, length + ans);
    }

    public static int index(char c) {
        return c - 'a' + 1;
    }

    /**
     * 给你一个下标从0开始的字符串数组words。每个字符串都只包含 小写英文字母。words中任意一个子串中，每个字母都至多只出现一次。
     *
     * 如果通过以下操作之一，我们可以从 s1的字母集合得到 s2的字母集合，那么我们称这两个字符串为 关联的：
     *
     * 往s1的字母集合中添加一个字母。
     * 从s1的字母集合中删去一个字母。
     * 将 s1中的一个字母替换成另外任意一个字母（也可以替换为这个字母本身）。
     * 数组words可以分为一个或者多个无交集的 组。如果一个字符串与另一个字符串关联，那么它们应当属于同一个组。
     *
     * 注意，你需要确保分好组后，一个组内的任一字符串与其他组的字符串都不关联。可以证明在这个条件下，分组方案是唯一的。
     *
     * 请你返回一个长度为 2的数组ans：
     *
     * ans[0]是words分组后的总组数。
     * ans[1]是字符串数目最多的组所包含的字符串数目。
     *
     * 来源：力扣（LeetCode）
     * 链接：https://leetcode-cn.com/problems/groups-of-strings
     * 著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    public static int[] groupStrings(String[] words) {
        if (null == words || words.length == 0) {
            return null;
        }
        int n = words.length;
        HashMap<Integer, Integer> map = new HashMap<>();
        for (String word : words) {
            int mask = 0;
            for (int i = 0; i < word.length(); i++) {
                mask |= (1 << (word.charAt(i) - 'a'));
            }
            int val = map.getOrDefault(mask, 0);
            map.put(mask, val + 1);
        }

        HashSet<Integer> visited = new HashSet<>();
        int groupNum = 0;
        int maxNumPerGroup = 0;
        for (int key : map.keySet()) {
            if (!visited.contains(key)) {
                groupNum++;
                maxNumPerGroup = Math.max(maxNumPerGroup, dfs(map, key, visited));
            }
        }
        return new int[]{groupNum, maxNumPerGroup};
    }

    private static int dfs(HashMap<Integer, Integer> map, int key, HashSet<Integer> visited) {
        int num = 0;
        List<Integer> adj = getAdj(key);
        for (int i = 0; i < adj.size(); i++) {
            int adjKey = adj.get(i);
            if (map.containsKey(adjKey) && !visited.contains(adjKey)) {
                num += map.get(adjKey);
                visited.add(adjKey);
                num += dfs(map, adjKey, visited);
            }
        }
        return num;
    }

    private static List<Integer> getAdj(int key) {
        List<Integer> list = new ArrayList<>();
        for (int i = 0; i < 26; i++) {
            list.add(key ^ (1 << i));
        }
        for (int i = 0; i < 26; i++) {
            if (((key >> i) & 1) == 1) {
                int tmp = (key ^ (1 << i));
                for (int j = 0; j < 26; j++) {
                    if (((tmp >> j) & 1) == 0) {
                        list.add(tmp ^ (1 << j));
                    }
                }
            }
        }
        return list;
    }
}
